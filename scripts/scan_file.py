import os

from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pandas as pd
from scipy import signal

import torch

from whale_gunshot_localization.models.tcn_archs import TCNRangeAndClassify
from whale_gunshot_localization.utils.experimental import ClipAnalyzer
from whale_gunshot_localization import config, PROJECT_ROOT_DIR

# config
FILE = os.path.join(config['dataset']['ccb_data_directory'], '6470/6470.220406001610.wav')
OVERLAP_FRACTION = 0.75 # in (0, 1)
CHUNK_SIZE = 160 # [s]

# set constants
mu_list = np.load(config['dataset']['data_directory'] + '/mean_range_classification.npy', allow_pickle=True)
std_list = np.load(config['dataset']['data_directory'] + '/std_range_classification.npy', allow_pickle=True)
fs = config['signal']['fs']
T = config['signal']['T']
size = fs*T

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!\n")
else:
    print("WARNING: Could not find GPU. Using CPU only.\n")

# make results CSV file if it doesn't exist
if not os.path.exists(os.path.join(PROJECT_ROOT_DIR, 'scan_results.csv')):
    pd.DataFrame(columns=["file_name", "timestamp"]).to_csv(os.path.join(PROJECT_ROOT_DIR, 'scan_results.csv'), index=False)

# load model
model = TCNRangeAndClassify(input_size=232,
                            num_channels=7*[368],
                            kernel_size=6,
                            dropout=0.05).to(device)
state_dict = torch.load(os.path.join(config['models']['checkpoints_directories'] + '/range_classify', 'weights_best.pt'), map_location=device)
model.load_state_dict(state_dict['model_state_dict'])

# clip analyzer
CA = ClipAnalyzer(model, mu_list, std_list, fs, T, device, overlap_fraction=OVERLAP_FRACTION)

# get file duration and sample rate
file_duration = librosa.get_duration(filename=FILE)
file_samplerate = librosa.get_samplerate(path=FILE)

pointer = 0
c = 0
with tqdm(total=file_duration // CHUNK_SIZE) as pbar:
    while file_duration - pointer > CHUNK_SIZE:

        # get CHUNK of data and downsample
        y, _ = librosa.load(FILE, sr=file_samplerate, offset=pointer, duration=CHUNK_SIZE)
        y = signal.resample_poly(y, fs, file_samplerate)
        
        # scan CHUNK
        imgs, range_measurements, timestamps = CA.process_clips(y)[0]

        if len(timestamps):
            new_row = pd.DataFrame({'file_name' : [FILE for _ in timestamps], 'timestamp' : [pointer + t for t in timestamps]})
            new_row.to_csv('scan_results.csv', mode='a', index=False, header=False)
            
            for i in imgs:
                plt.imshow(i)
                plt.savefig(f'{c}.png')
                c += 1

        pointer += CHUNK_SIZE
        pbar.update(1)