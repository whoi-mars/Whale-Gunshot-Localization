"""
Script to scan a file of experimental data.
"""

import os
import argparse
import sys
import warnings

from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path

import torch

from whale_gunshot_localization.models.tcn_archs import TCNRangeAndClassify
from whale_gunshot_localization.utils.experimental import ClipAnalyzer, l2_standardize, get_wav_timestamp
from whale_gunshot_localization import config, PROJECT_ROOT_DIR

def scan_file(file, data_params, overlap_fraction, chunk_size, model, device, background=False):

    """
    Scan single file for gunshots using a trained TCN.

    Parameters
    ----------
    file : str
        path to WAV file to scan
    data_params : dict
        dictionary containing the following key-value pairs:
        mu_list : array-like
            mean for each spectrogram row across training set
        std_list : array-like
            std for each spectrogram row across training set
        fs : float
            sampling frequency
        T : float
            signal duration
    overlap_fraction : float in (0, 1)
        how much window to overlap when scanning the file
    chunk_size : float
        chunk of WAV file to process simultaneously with the TCN in seconds
    model : torch.nn.module
        Pytorch model
    device : torch.device
        device to run the model on
    background : bool
        silence the progress bar
    """

    assert set(data_params.keys()) == set(['mu_list', 'std_list', 'fs', 'T']), 'check items in data_params dictionary.'

    # set up results directory
    results_path = os.path.join(PROJECT_ROOT_DIR, "scripts", "results")
    Path(results_path).mkdir(exist_ok=True)

    # path for results
    csv_path = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", "scan_results.csv")
    sensor = file.split('/')[-2]

    # make results CSV file if it doesn't exist
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["sensor", "file_name", "timestamp", "global_timestamp", "range"]).to_csv(csv_path, index=False)

    # clip analyzer
    CA = ClipAnalyzer(model, 
                      data_params,
                      preprocessor=l2_standardize,
                      device=device, 
                      overlap_fraction=overlap_fraction)

    # get file duration and sample rate
    file_duration = librosa.get_duration(filename=file)
    file_samplerate = librosa.get_samplerate(path=file)
    
    # return if nothing to scan
    if file_duration < data_params['T']:
        warnings.warn(f"The duration of {file} is shorter than the considered signal duration.")
        sys.stderr.flush()
        return

    pointer = 0
    # c = 0
    with tqdm(total=file_duration // (chunk_size - (overlap_fraction * data_params['T'])), disable=background) as pbar:
        while file_duration - pointer > chunk_size:

            # get CHUNK of data and downsample
            y, _ = librosa.load(file, sr=file_samplerate, offset=pointer, duration=chunk_size)
            y = signal.resample_poly(y, data_params['fs'], file_samplerate)
            
            # scan CHUNK
            imgs, range_measurements, timestamps = CA.process_clips(y)[0]

            if len(timestamps):
                row_dict['sensor'] = [sensor for _ in timestamps]
                row_dict['file_name'] = [file for _ in timestamps]
                row_dict['timestamp'] = [pointer + t for t in timestamps]
                row_dict['global_timestamp'] = [get_wav_timestamp(f, int((ts) * 1000)) for f, ts in zip(row_dict['file_name'], row_dict['timestamp'])]
                row_dict['rnage'] = [range_measurement for range_measurement in range_measurements]
                new_row = pd.DataFrame(row_dict)
                new_row.to_csv(csv_path, mode='a', index=False, header=False)
                
                # for i in imgs:
                #     plt.imshow(i)
                #     plt.savefig(os.path.join(results_path, f'{c}.png'))
                #     c += 1

            pointer += (chunk_size - (overlap_fraction * data_params['T']))
            pbar.update(1)

if __name__ == "__main__":

    # parse input arguments
    parser = argparse.ArgumentParser(description="Scan a single WAV file for gunshots using a trained TCN.")
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='path to WAV file to scan, relative to the root directory with acoustic data')
    parser.add_argument('-o', '--overlap_fraction', type=float, default=0.75, metavar="[float in (0, 1)]", required=False,
                        help='how much window to overlap when scanning the file (default: 0.75)')
    parser.add_argument('-c', '--chunk_size', type=float, default=160, required=False,
                        help='chunk of WAV file to process simultaneously with the TCN in seconds (default: 160)')

    # parse arguments
    args = parser.parse_args()
    assert 0 < args.overlap_fraction < 1, 'overlap_fraction must be in (0, 1).'
    data_params = {'mu_list' : np.load(config['dataset']['data_directory'] + '/mean_range_classification.npy', allow_pickle=True),
                   'std_list' : np.load(config['dataset']['data_directory'] + '/std_range_classification.npy', allow_pickle=True),
                   'fs' : config['signal']['fs'],
                   'T' : config['signal']['T'],}

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
    
    # scan file
    scan_file(file=os.path.join(config['dataset']['ccb_data_directory'], args.file),
              data_params=data_params,
              overlap_fraction=args.overlap_fraction, 
              chunk_size=args.chunk_size, 
              model=model,
              device=device)
    