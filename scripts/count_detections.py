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
from whale_gunshot_localization.utils.experimental import ClipAnalyzer, l2_standardize
from whale_gunshot_localization import config, PROJECT_ROOT_DIR
from scan_file import scan_file

parser = argparse.ArgumentParser(description="Scan all files in sensor.")
parser.add_argument('--scan', action='store_true',
                    help='whether to scan before plotting')
parser.add_argument('--sensor', '-s', type=int, required=True,
                    help='which sensor to scan')
parser.add_argument('-o', '--overlap_fraction', type=float, default=0.75, metavar="[float in (0, 1)]", required=False,
                    help='how much window to overlap when scanning the file (default: 0.75)')
parser.add_argument('-c', '--chunk_size', type=float, default=160, required=False,
                    help='chunk of WAV file to process simultaneously with the TCN in seconds (default: 160)')
parser.add_argument('--background', '-b', action='store_true',
                    help='silence the progress bar')
args = parser.parse_args()

def sort_wav_chronological(wav_files, xml_files):

    start_times = []
    end_times = []

    for wf in wav_files:

        # get corresponding xml file
        xml_file = wf.split('.wav')[0] + ".log.xml"

        # get start time
        with open(xml_file, 'r') as f:
            data = f.read()
        data = BeautifulSoup(data, features='lxml')
        start_time = np.datetime64(data.find_all("wavfilehandler", samplingstarttimelocal=True)[0]['samplingstarttimelocal'])
        end_time = np.datetime64(data.find_all("wavfilehandler", samplingstoptimelocal=True)[0]['samplingstoptimelocal'])
        start_times.append(start_time)
        end_times.append(end_time)

    # get sorted indices
    idx_sorted = np.argsort(start_times)

    start_times = np.asarray(start_times)
    end_times = np.asarray(end_times)
    
    return wav_files[idx_sorted], start_times[idx_sorted], end_times[idx_sorted]


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

        # get files
        wav_files = np.asarray(glob.glob(os.path.join(config['dataset']['ccb_data_directory'], str(args.sensor), "*.wav")))
        xml_files = np.asarray(glob.glob(os.path.join(config['dataset']['ccb_data_directory'], str(args.sensor), "*.xml")))

        # sort wav files chronologically
        wav_files, start_times, end_times = sort_wav_chronological(wav_files, xml_files)

        data_params = {'mu_list' : np.load(config['dataset']['data_directory'] + '/mean_range_classification.npy', allow_pickle=True),
                    'std_list' : np.load(config['dataset']['data_directory'] + '/std_range_classification.npy', allow_pickle=True),
                    'fs' : config['signal']['fs'],
                    'T' : config['signal']['T'],}

        # scan file
        total = len(wav_files)
        for i, wav_file in enumerate(wav_files):

            # if running as a background process just print progress
            if args.background:
                print(f'{i+1}/{total}', flush=True)
            
            scan_file(file=wav_file,
                        data_params=data_params,
                        overlap_fraction=args.overlap_fraction, 
                        chunk_size=args.chunk_size, 
                        model=model,
                        device=device,
                        background=args.background)
            print()
    
    # results path
    csv_path = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", "scan_results.csv")
    img_path = os.path.join(PROJECT_ROOT_DIR, "scripts", "results")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise RuntimeError(f'{csv_path} does not exist, you must scan the data before plotting')

    # get sensors in results CSV
    sensors = list(set(df["sensor"]))

    if args.sensor not in sensors:
        raise ValueError(f"no scanned files from sensor {args.sensor}")
    
    # make dictionary from file --> start time
    wav_files, xml_files = [], []
    for sensor in sensors:
        wav_files.extend(glob.glob(os.path.join(config['dataset']['ccb_data_directory'], str(sensor), "*.wav")))
        xml_files.extend(glob.glob(os.path.join(config['dataset']['ccb_data_directory'], str(sensor), "*.xml")))
    wav_files = np.asarray(wav_files)
    xml_files = np.asarray(xml_files)
    wav_files, start_times, end_times = sort_wav_chronological(wav_files, xml_files)
    start_time_dict = dict(zip(wav_files, start_times))
    end_time_dict = dict(zip(wav_files, end_times))

    files = list(set(df['file_name']))
    fig_all, ax_all = plt.subplots(len(files), 1)
    fig_all.subplots_adjust(hspace=0.5)
    for i, f in enumerate(files):
        
        # filter by chosen sensor and current file
        df_file = df[(df['sensor'] == args.sensor) & (df['file_name'] == f)].copy()

        # add column with timestamp relative to global time
        df_file['global_timestamp'] = df_file.apply(lambda row : start_time_dict[row['file_name']] + np.timedelta64(int(row['timestamp'] * 1000), 'ms'), axis=1)
        
        # get bin edges
        bins_dt = pd.date_range(start=start_time_dict[f], end=end_time_dict[f], freq="2min")
        
        # bin labels are left edge
        # bins_str = bins_dt.astype(str).values
        # labels = [bins_dt[i-1] for i in range(1, len(bins_str))]
        
        hist_vals = pd.to_datetime(pd.cut(df_file["global_timestamp"], bins=bins_dt, labels=bins_dt[:-1]).dropna())
        hist, bins = np.histogram(hist_vals, bins=bins_dt)
        
        # get x limits for plots
        x_lims = mdates.date2num(bins_dt)

        # add to aggregate plot
        ax_all[i].imshow(hist[np.newaxis,:], cmap="plasma", aspect="auto", extent=[x_lims[0], x_lims[-1], 0, 1])
        ax_all[i].set_yticks([])
        ax_all[i].set_xticks([])

        # file figure
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(hist[np.newaxis,:], cmap="plasma", aspect="auto", extent=[x_lims[0], x_lims[-1], 0, 1])
        plt.colorbar(im, ax=ax, label="Detection Count")
        ax.xaxis_date()
        date_format = mdates.DateFormatter('%D -- %H:%M:%S')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_title(f.split('/')[-1])
        fig.autofmt_xdate()
        fig.savefig(os.path.join(img_path, f.split('/')[-1].split('.wav')[0] + ".png"))
        plt.close(fig)

    fig_all.savefig(os.path.join(img_path, 'all_files.png'))
    plt.close(fig_all)