import os
import glob
import argparse

import torch
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from whale_gunshot_localization.models.tcn_archs import TCNRangeAndClassify
from whale_gunshot_localization.utils.experimental import ClipAnalyzer, l2_standardize, sort_wav_chronological
from whale_gunshot_localization import config, PROJECT_ROOT_DIR
from scan_file import scan_file

parser = argparse.ArgumentParser(description="Scan all files in sensor.")
parser.add_argument('--scan', action='store_true',
                    help='whether to scan before plotting')
parser.add_argument('--sensor', '-s', type=int, nargs='+', default=config['TOSSIT']['ids'],
                    help='which sensor to scan')
parser.add_argument('-o', '--overlap_fraction', type=float, default=0.75, metavar="[float in (0, 1)]", required=False,
                    help='how much window to overlap when scanning the file (default: 0.75)')
parser.add_argument('-c', '--chunk_size', type=float, default=160, required=False,
                    help='chunk of WAV file to process simultaneously with the TCN in seconds (default: 160)')
parser.add_argument('--background', '-b', action='store_true',
                    help='silence the progress bar')
args = parser.parse_args()

if __name__ == "__main__":
    
    if args.scan:
        # get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using the GPU!\n")
        else:
            print("WARNING: Could not find GPU. Using CPU only.\n")

        # load model
        model = TCNRangeAndClassify(input_size=232,
                                    num_channels=7*[368],
                                    kernel_size=6,
                                    dropout=0.05).to(device)
        state_dict = torch.load(os.path.join(config['models']['checkpoints_directories'] + '/range_classify', 'weights_best.pt'), map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])

        for idx, sensor in enumerate(args.sensor):
            print(f"sensor: {sensor} -- ({idx+1}/{len(args.sensor)})\n")

            # get files
            wav_files = np.asarray(glob.glob(os.path.join(config['dataset']['ccb_data_directory'], str(sensor), "*.wav")))

            # sort wav files chronologically
            wav_files, start_times, end_times = sort_wav_chronological(wav_files)

            data_params = {'mu_list' : np.load(config['dataset']['data_directory'] + '/mean_range_classification.npy', allow_pickle=True),
                        'std_list' : np.load(config['dataset']['data_directory'] + '/std_range_classification.npy', allow_pickle=True),
                        'fs' : config['signal']['fs'],
                        'T' : config['signal']['T'],}

            # scan file
            total = len(wav_files)
            for i, wav_file in enumerate(wav_files):

                # if running as a background process just print progress
                print(f'{i+1}/{total}', flush=True)
                
                scan_file(file=wav_file,
                          data_params=data_params,
                          overlap_fraction=args.overlap_fraction, 
                          chunk_size=args.chunk_size, 
                          model=model,
                          device=device,
                          background=args.background)
                print()
    
    # results paths
    csv_path = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", "scan_results.csv")
    img_path = os.path.join(PROJECT_ROOT_DIR, "scripts", "results")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise RuntimeError(f'{csv_path} does not exist, you must scan the data before plotting')

    # get sensors in results CSV
    sensors = list(set(df["sensor"]))

    # check that desired sensors are in the saved data
    if any(int(s) not in sensors for s in args.sensor):
        raise ValueError(f"no scanned files from sensor {args.sensor}")

    # make dictionary from file --> start time
    wav_files = []
    for sensor in sensors:
        wav_files.extend(glob.glob(os.path.join(config['dataset']['ccb_data_directory'], str(sensor), "*.wav")))
    wav_files = np.asarray(wav_files)
    wav_files, start_times, end_times = sort_wav_chronological(wav_files)
    start_time_dict = dict(zip(wav_files, start_times))
    end_time_dict = dict(zip(wav_files, end_times))

    # add global timestamp to df
    df['global_timestamp'] = df.apply(lambda row : start_time_dict[row['file_name']] + np.timedelta64(int(row['timestamp'] * 1000), 'ms'), axis=1)
    
    # bins each detection
    bins = pd.date_range(start=df['global_timestamp'].min().date(), end=df['global_timestamp'].max().date() + pd.Timedelta(1, "d"), freq="D")
    df["bin"] = pd.to_datetime(pd.cut(df["global_timestamp"], bins=bins, labels=bins[:-1]))
    
    # create plots of detections for each day
    for day in bins[:-1]:

        # bins to chunks the day's detections
        day_bins = pd.date_range(start=day, end=day + pd.Timedelta(1, "d"), freq="2min")

        histograms = []
        for sensor in args.sensor:

            # get df for a particular sensor on a particular day
            sub_df = df[(df["sensor"] == int(sensor)) & (df["bin"] == day)]

            # calculate historgram
            hist_vals = pd.cut(sub_df["global_timestamp"], bins=day_bins, labels=np.arange(len(day_bins[:-1])) + 1)
            hist, _ = np.histogram(hist_vals, bins=np.arange(len(day_bins[:-1])) + 0.5)
            histograms.append(hist)

        # get max bin count to calibrate colorbar, and skip day if max bin count is 0
        max_count = max(max(hist) for hist in histograms)
        if max_count == 0:
            continue
        
        # make day plot
        fig, axs = plt.subplots(len(args.sensor), 1, sharex=True)
        x_lims = mdates.date2num(day_bins)

        for hist, ax, sensor in zip(histograms, axs, args.sensor):
            im = ax.imshow(hist[np.newaxis,:], cmap="plasma", aspect="auto", extent=[x_lims[0], x_lims[-1], 0, 1], vmin=0, vmax=max_count)
            ax.set_title(f"sensor {sensor}")
            ax.set_yticks([])

        axs[0].xaxis_date()
        date_format = mdates.DateFormatter('%H:%M:%S')
        axs[0].xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        fig.suptitle(str(day.date()), fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.86, 0.16, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Counts")
        fig.savefig(os.path.join(img_path, f"{day.date()}.png"))
