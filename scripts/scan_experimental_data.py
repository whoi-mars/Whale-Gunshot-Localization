"""
Script to scan multiple sensors of experimental data.
"""

import os
import glob
import yaml
import pickle
import copy
import argparse
import warnings
import sys
import datetime
from collections import Counter

from tqdm import tqdm
import librosa
import numpy as np
import torch
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import pyproj as proj
import pygmt

from whale_gunshot_localization.models.tcn_archs import TCNRangeAndClassify
import whale_gunshot_localization.utils.experimental as experimental
from whale_gunshot_localization import config, PROJECT_ROOT_DIR

# parse input arguments
parser = argparse.ArgumentParser(description="Scan group of corresponding wave files")
parser.add_argument('--start', type=lambda ts : datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'), default=None,
                    help='start timestamp of scan')
parser.add_argument('--end', type=lambda ts : datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'), default=None,
                    help='start timestamp of scan')
parser.add_argument('--ordered_sensors', nargs='+', type=str, default=config['TOSSIT']['ids'],
                    help='sensor IDs in the same order as the sensor positions in config.TOSSIT_locations')
parser.add_argument('--half_window', type=float, default=80,
                    help='size of the data window to analyze at once after a course synchronization has been attempted')
parser.add_argument('-o', '--overlap_fraction', type=float, default=0.75, metavar="[float in (0, 1)]", required=False,
                    help='how much window to overlap when scanning the file (default: 0.75)')
parser.add_argument('--scan', action='store_true',
                    help='whether to scan or just plot')
parser.add_argument('--background', '-b', action='store_true',
                    help='silence the progress bar')
parser.add_argument('--suppress_warnings', action='store_true',
                    help="tell Python to suppress warnings")
args = parser.parse_args()

# suppress warnings
if args.suppress_warnings:
    import warnings
    warnings.filterwarnings("ignore")

if args.scan:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!", flush=True)
    else:
        print("WARNING: Could not find GPU. Using CPU only.", flush=True)

def has_len(x):
    """
    Checks if an array is empty    
    """
    return len(x) > 0

def scan_experimental_data(model, start, end, chunk_size, overlap_fraction, ordered_sensors, background=False):
    """
    Given a WAV file from a sensor, find the corresponding WAV files from the othe files (if they exist)
    and perform detection and subsequently data assocaition and localization. Results are saved in a CSV file.

    Parameters
    ----------
    model : torch.nn.module
        Pytorch model
    file : str
        path to WAV file to find matches for and scan
    chunk_size : float
        chunk of WAV file to process simultaneously with the TCN in seconds
    overlap_fraction : float in (0, 1)
        how much window to overlap when scanning the file
    ordered_sensors : List[str]
        list of sensor ids in the order in which the sensors are stored in the config file
    background : bool
        silence the progress bar
    """
    
    # set up results directory
    Path(os.path.join(PROJECT_ROOT_DIR, "scripts", "results", config['models']['model_dir'])).mkdir(parents=True, exist_ok=True)

    # cast ordered sensors as numpy
    ordered_sensors = np.asarray(ordered_sensors)

    # get files associated with all sensors
    wav_files = []
    for s in ordered_sensors:
        wav_files.extend(glob.glob(os.path.join(config['dataset']['ccb_data_directory'], s, "*.wav")))
    wav_files = np.asarray(wav_files)
    
    # sort wav files in chonological order by start time
    wav_files, start_times, end_times = experimental.sort_wav_chronological(wav_files)
    start_time_dict = dict(zip(wav_files, start_times))
    end_time_dict = dict(zip(wav_files, end_times))

    # object to read timestamp from WAV file
    wr = experimental.WAVReader(sensors=ordered_sensors, chunk_size=chunk_size, fs_desired=config['signal']['fs'])

    # set up CSV for saving results
    csv_path = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", config['models']['model_dir'], "multi_scan_results.csv")
    columns = ["id", "sensor", "file_name", "timestamp", "global_timestamp", "range", "y", "x"]  
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

    # get next id to write
    df = pd.read_csv(csv_path)
    id = 1 if np.isnan(df["id"].max()) else df["id"].max() + 1

    ############################################################
    #                     commence scanning                    #
    ############################################################

    # prepare scanner object
    data_params = {'mu_list' : np.load(config['dataset']['data_directory'] + '/mean_range_classification.npy', allow_pickle=True),
                'std_list' : np.load(config['dataset']['data_directory'] + '/std_range_classification.npy', allow_pickle=True),
                'fs' : config['signal']['fs'],
                'T' : config['signal']['T'],}
    CA = experimental.ClipAnalyzer(model, data_params, preprocessor=experimental.l2_standardize, device=device)

    # prepare localizer object
    l = experimental.Localizer(k=4, multilat=experimental.MultilaterationOpt(method_thresh=0.95), consistency_thresh=500, prune=False)

    # get starts of chunks to read and total days
    chunk_starts = pd.date_range(start=start, end=end, freq=f"{chunk_size}s")
    chunk_starts = chunk_starts[chunk_starts < (end - datetime.timedelta(seconds=chunk_size))]
    total_days = (end - start) / datetime.timedelta(days=1)

    # get count of chunks which start on each day
    day_counts = Counter([date.date() for date in chunk_starts])

    # scan files
    day_counter = 1
    curr_date = chunk_starts[0].date()
    if background:
        print(f"day {day_counter}/{total_days}\n", flush=True)
    with tqdm(total=day_counts[curr_date], disable=background) as pbar:
        for i, start_time in enumerate(chunk_starts):

            # if we are on a new day, reset tqdm bar
            if start_time.date() != curr_date:
                curr_date = start_time.date()
                day_counter += 1
                pbar.reset(total=day_counts[curr_date])
                pbar.set_postfix({'day': day_counter, 'total': total_days})
                if background:
                    print(f"day {day_counter}/{total_days}\n", flush=True)

            # get data
            success, file_list, data = wr.get_audio(timestamp=start_time.to_numpy().astype('datetime64[s]'))

            if success:
                # scan clips
                res = CA.process_clips(data)
                
                # format outputs
                ranges = [x[1] for x in res]
                timestamps = [x[2] for x in res]

                # check if enough measurements have been detected across all sensors
                if len(list(filter(has_len, ranges))) >= 4:
                    
                    # build hypergraph
                    success = l.set_measurements(ranges, adaptive=False)

                    if success:
                        # assocaite/localize
                        assocs, locs_est = l.associate_and_localize(reduce_dups=False, last_step=True)

                        # flatten outputs
                        ranges_flat, timestamps_flat = [], []
                        sensor_map = {}
                        linear_counter = 0
                        for i, (r, t) in enumerate(zip(ranges, timestamps)):
                            for j, (rr, tt) in enumerate(zip(r, t)):
                                
                                ranges_flat.append(rr)
                                timestamps_flat.append(tt)
                                sensor_map[linear_counter] = i

                                linear_counter += 1
                        
                        # create dictionary to save results
                        row_dict = {col : [] for col in columns}
                        for i, assoc in enumerate(assocs):
                            if locs_est[i,0] >= config['scaling']['min_y'] \
                               and locs_est[i,0] <= config['scaling']['max_y'] \
                               and locs_est[i,1] >= config['scaling']['min_x'] \
                               and locs_est[i,1] <= config['scaling']['max_x']:
                                for m in assoc:
                                    row_dict["id"].append(id)
                                    row_dict["sensor"].append(ordered_sensors[sensor_map[m]])
                                    row_dict["file_name"].append(file_list[sensor_map[m]])
                                    row_dict["timestamp"].append(((start_time.to_numpy() + np.timedelta64(int(timestamps_flat[m] * 1000), "ms")) - start_time_dict[row_dict["file_name"][-1]]).astype('timedelta64[s]').astype(float))
                                    row_dict["global_timestamp"].append(experimental.get_wav_timestamp(row_dict["file_name"][-1], int((row_dict["timestamp"][-1]) * 1000)))
                                    row_dict["range"].append(np.around(ranges_flat[m], 2))
                                    row_dict["x"].append(np.around(locs_est[i,1], 2))
                                    row_dict["y"].append(np.around(locs_est[i,0], 2))
                                id += 1

                        new_row = pd.DataFrame(row_dict)
                        new_row.to_csv(csv_path, mode='a', index=False, header=False)
                    # update pointer and proress bar
                    l.reset()
            pbar.update(1)

if __name__ == '__main__':

    ###################################
    #            scan data            #
    ###################################

    # get files associated with first sensor in ordered_sensors list
    wav_files = []
    for s in args.ordered_sensors:
        wav_files.extend(glob.glob(os.path.join(config['dataset']['ccb_data_directory'], s, "*.wav")))
    wav_files = np.asarray(wav_files)

    # sort wav files in chonological order by start time
    wav_files, start_times, end_times = experimental.sort_wav_chronological(wav_files)
    start_time_dict = dict(zip(wav_files, start_times))
    end_time_dict = dict(zip(wav_files, end_times))

    # properly set start and end times
    if args.start is None:
        args.start = np.asarray(list(start_time_dict.values())).min().astype(datetime.datetime).replace(hour=0, minute=0, second=0)
        args.end = np.asarray(list(end_time_dict.values())).max().astype(datetime.datetime).replace(hour=0, minute=0, second=0)
    elif args.end is None:
        args.end = (args.start + datetime.timedelta(days=1))

    if args.scan:
        # load model
        model = TCNRangeAndClassify(input_size=232, 
                                    num_channels=7*[368],
                                    kernel_size=6,
                                    dropout=0.05).to(device)

        # load model weights
        state_dict = torch.load(os.path.join(config['models']['checkpoints_directories'], config['models']['model_dir'], 'weights_best.pt'),
                                map_location=device)['model_state_dict']
        model.load_state_dict(state_dict)

        # perform scanning
        scan_experimental_data(model, args.start, args.end, 2 * args.half_window, args.overlap_fraction, args.ordered_sensors, args.background)

    # ###################################
    # #           plot results          #
    # ###################################

    # fig_dir = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", config['models']['model_dir'], "detection_maps")
    # Path(fig_dir).mkdir(exist_ok=True, parents=True)
    
    # csv_path = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", config['models']['model_dir'], "multi_scan_results.csv")
    # if not os.path.exists(csv_path):
    #     raise IOError("no results file")
    # df = pd.read_csv(csv_path)

    # # get bin edges
    # bins_dt = pd.date_range(start=pd.to_datetime(start_time_dict[wav_files[0]]).date(), end=(pd.to_datetime(end_time_dict[wav_files[-1]]) + pd.Timedelta(1, "d")).date(), freq="D")
    # df["bin"] = pd.to_datetime(pd.cut(pd.DatetimeIndex(df["global_timestamp"]), bins=bins_dt, labels=bins_dt[:-1]))
    # bin_grouped = df.groupby(by="bin")

    # # # load map
    # # Image.MAX_IMAGE_PIXELS = 729744000
    # # bathym = Image.open(os.path.join(config['dataset']['data_directory'], "mikesbathym.tif"))

    # # make plots
    # all_locs_est = []
    # all_dates = []
    # for _, dfg_bin in bin_grouped:
    #     id_grouped = dfg_bin.groupby(by="id")

    #     x, y = [],[]
    #     date = None 
    #     for _, dfg in id_grouped:
    #         dfg = dfg.reset_index()
    #         x.append(dfg.loc[0,"x"])
    #         y.append(dfg.loc[0,"y"])
    #         if date is None:
    #             date = dfg.loc[0,"bin"]

    #     # plot locations by day
    #     locs_est = np.stack([y, x], axis=1)
    #     plotting.plot_localization(locs_est, title="CCB-2023 Location Estimates", save=os.path.join(fig_dir, f"locations_{date.date()}.png"), dates=date.date(), bins=10, d_lat=0.2, d_lon=0.1)
    #     all_dates.append(date.date())
    #     all_locs_est.append(locs_est)
        
    # # plot all days
    # plotting.plot_localization(all_locs_est, title="CCB-2023 Location Estimates", save=os.path.join(fig_dir, f"locations_all_dates.png"), dates=all_dates, bins=10, d_lat=0.2, d_lon=0.1)
