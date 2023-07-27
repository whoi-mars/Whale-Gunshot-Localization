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

from tqdm import tqdm
import librosa
import numpy as np
import torch
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

from whale_gunshot_localization.models.tcn_archs import TCNRangeAndClassify
import whale_gunshot_localization.utils.experimental as experimental
from whale_gunshot_localization import config, PROJECT_ROOT_DIR

# parse input arguments
parser = argparse.ArgumentParser(description="Scan group of corresponding wave files")
parser.add_argument('-f', '--file', type=str, default=None,
                    help='path to WAV file to be used to fetch corresponding WAV files from other sensors to scan')
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
args = parser.parse_args()

if args.scan:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU. Using CPU only.")

def plot_localization(locs_est, buffer=13000, title=None, save=None): 
    """
    Plot estimated source locations.

    Parameters
    ----------
    locs_est : np.array, of shape N X 2
        list of estimated locations where the first column stores the Y
        coordinate and the second stores the X coordinates
    buffer : float
        how much to plot outside of the limits established in the config file
    title : str
        plot title
    save : str
        path at which to save the plot if desired
    """

    # load constants
    TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
    min_x = config['scaling']['min_x']
    max_x = config['scaling']['max_x']
    min_y = config['scaling']['min_y']
    max_y = config['scaling']['max_y']

    plt.plot(TOSSIT_locations[:,1], TOSSIT_locations[:,0], '^', markersize=12, label='sensor', markeredgewidth=2)
    plt.plot(locs_est[:,1], locs_est[:,0], 'gx', label='estimate', markersize=4, markeredgewidth=1)
    if title is not None:
        plt.title(title)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.xlim([min_x-buffer, max_x+buffer])
    plt.ylim([min_y-buffer, max_y+buffer])
    plt.gca().invert_yaxis()
    plt.legend()
    
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()

def has_len(x):
    """
    Checks if an array is empty    
    """
    return len(x) > 0

def scan_experimental_data(model, file, chunk_size, overlap_fraction, ordered_sensors, background=False):
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

    # cast ordered sensors as numpy
    ordered_sensors = np.asarray(ordered_sensors)

    # if we already have the matching files stored, load them. if not, do the matching.
    if os.path.exists(os.path.join(config['dataset']['ccb_data_directory'], "matching files.p")):
        print("Found matched TOSSIT files...", flush=True)
        with open(os.path.join(config['dataset']['ccb_data_directory'], "matching files.p"), "rb") as f:
            matching_files = pickle.load(f)
    else:
        print("Matching corresponding files from each TOSSIT...", flush=True)
        matching_files = experimental.get_matching_files(TOSSIT_dirs=ordered_sensors)
        with open(os.path.join(config['dataset']['ccb_data_directory'], "matching files.p"), "wb") as f:
            pickle.dump(copy.deepcopy(matching_files), f)

    # get order of sensors in 'matching_files.p'
    match_sensor_order = []
    for s in matching_files[0]:
        id = str(s[0].split('/')[-2])
        match_sensor_order.extend(np.argwhere(ordered_sensors == id)[0])
    match_sensor_order = np.asarray(match_sensor_order)
    
    # search for the match set with the desired file
    target_file = os.path.join(config['dataset']['ccb_data_directory'], file)
    idx = np.argwhere(ordered_sensors[match_sensor_order] == target_file.split('/')[-2])[0][0]

    target_group = None
    for group in matching_files:
        if group[idx][0] == target_file:
            target_group = group
            break
    if target_group is None:
        warnings.warn(f"{target_file} does not exist or is not a part of a matching group")
        sys.stderr.flush()
        return

    # sort target group to be in correct sensor order and split into files and dts
    target_group = [x[0] for x in sorted(zip(target_group, match_sensor_order), key=lambda pair : pair[1])]
    file_list, dt_list = [], []
    for sensor in target_group:
        path, dt = sensor
        file_list.append(path)
        dt_list.append(dt)
    file_list = np.asarray(file_list)
    dt_list = np.asarray(dt_list)

    # adjust dt_list so that we don't need to go back in time for any sensor.
    # in this way, we can start as close to the beginning of all files as possible after a
    # rough synchronization
    dt_list = dt_list - np.min(dt_list[dt_list <= 0])

    # get shortest file duration to limit the total scan
    min_duration = min([librosa.get_duration(filename=f) - dt for f, dt in zip(file_list, dt_list)])

    # set up CSV for saving results
    csv_path = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", "multi_scan_results.csv")
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
    l = experimental.Localizer(k=4, multilat=experimental.MultilaterationOpt(method_thresh=0.95), consistency_thresh=1200, prune=False)

    # scan files
    pointer = 0
    with tqdm(total=min_duration // (chunk_size - (overlap_fraction * data_params['T'])), disable=background) as pbar:
        while min_duration - pointer > chunk_size: 
            
            # collect data
            data = []
            for f, dt in zip(file_list, dt_list):
                file_samplerate = librosa.get_samplerate(path=f)
                y, _ = librosa.load(f, sr=file_samplerate, offset=pointer + dt, duration=chunk_size)
                y = signal.resample_poly(y, data_params['fs'], file_samplerate)
                data.append(y)
            data = np.asarray(data)
            
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
                    assocs, locs_est = l.associate_and_localize(method='partition')
                    
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
                        for m in assoc:
                            row_dict["id"].append(id)
                            row_dict["sensor"].append(ordered_sensors[sensor_map[m]])
                            row_dict["file_name"].append(file_list[sensor_map[m]])
                            row_dict["timestamp"].append(pointer + dt_list[sensor_map[m]] + timestamps_flat[m])
                            row_dict["global_timestamp"].append(experimental.get_wav_timestamp(row_dict["file_name"][-1], int((row_dict["timestamp"][-1]) * 1000)))
                            row_dict["range"].append(np.around(ranges_flat[m], 2))
                            row_dict["x"].append(np.around(locs_est[i,1], 2))
                            row_dict["y"].append(np.around(locs_est[i,0], 2))
                        id += 1
                    
                    new_row = pd.DataFrame(row_dict)
                    new_row.to_csv(csv_path, mode='a', index=False, header=False)
            # update pointer and proress bar
            l.reset()
            pointer += (chunk_size - (overlap_fraction * data_params['T']))
            pbar.update(1)

if __name__ == '__main__':

    # get files associated with first sensor in ordered_sensors list
    wav_files = np.asarray(glob.glob(os.path.join(config['dataset']['ccb_data_directory'], args.ordered_sensors[0], "*.wav")))
    
    # sort wav files in chonological order by start time
    wav_files, start_times, end_times = experimental.sort_wav_chronological(wav_files)
    start_time_dict = dict(zip(wav_files, start_times))
    end_time_dict = dict(zip(wav_files, end_times))

    if args.scan:
        # load model
        model = TCNRangeAndClassify(input_size=232, 
                                    num_channels=7*[368],
                                    kernel_size=6,
                                    dropout=0.05).to(device)

        # load model weights
        state_dict = torch.load(os.path.join(config['models']['checkpoints_directories'] + config['models']['model_dir'], 'weights_best.pt'),
                                map_location=device)['model_state_dict']
        model.load_state_dict(state_dict)

        # perform scanning
        if args.file is None:
            # scan all wav files
            for i, wav in enumerate(wav_files):
                if args.background:
                    print()
                    print(f"{i+1}/{len(wav_files)}", flush=True)
                scan_experimental_data(model, os.path.join(*wav.split('/')[-2:]), 2 * args.half_window, args.overlap_fraction, args.ordered_sensors, args.background)
        else:
            scan_experimental_data(model, args.file, 2 * args.half_window, args.overlap_fraction, args.ordered_sensors, args.background)

    # plot results
    csv_path = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", "multi_scan_results.csv")
    df = pd.read_csv(csv_path)

    # get bin edges
    bins_dt = pd.date_range(start=pd.to_datetime(start_time_dict[wav_files[0]]).date(), end=(pd.to_datetime(end_time_dict[wav_files[-1]]) + pd.Timedelta(1, "d")).date(), freq="D")
    df["bin"] = pd.to_datetime(pd.cut(pd.DatetimeIndex(df["global_timestamp"]), bins=bins_dt, labels=bins_dt[:-1]))
    bin_grouped = df.groupby(by="bin")

    # get locations
    for _, dfg_bin in bin_grouped:
        id_grouped = dfg_bin.groupby(by="id")
        x, y = [],[]
        date = None 
        for _, dfg in id_grouped:
            dfg = dfg.reset_index()
            x.append(dfg.loc[0,"x"])
            y.append(dfg.loc[0,"y"])
            if date is None:
                date = dfg.loc[0,"bin"]
        # plot locations
        locs_est = np.stack([y, x], axis=1)
        plot_localization(locs_est, title=f"CCB-2022 Location Estimates - {date}", save=os.path.join(PROJECT_ROOT_DIR, "scripts", "results", f"locations_{date}.png"))
        plt.clf()
