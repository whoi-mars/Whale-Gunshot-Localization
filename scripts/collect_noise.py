"""
Script to collect noise to store in H5 file for training and in a MAT file for simulated data geneation.
"""

import os
import argparse

import hdf5storage
import h5py
import numpy as np
import librosa
from scipy.signal import resample_poly
from tqdm import tqdm

from whale_gunshot_localization import config, PROJECT_ROOT_DIR

# parse input arguments
parser = argparse.ArgumentParser(description="Collect examples randomly from experimental data to serve as noise-only examples for training.")
parser.add_argument('-n', '--number', type=int, default=10000,
                    help='number of examples to collect (default: 10000')
parser.add_argument('-fsmat', '--sample_rate_matlab', type=int, default=12000,
                    help='desired sample rate for MATLAB data simulation at which to save the collected examples (default: 12000 Hz)')
parser.add_argument('-fspy', '--sample_rate_python', type=int, default=600,
                    help='desired sample rate for NN training at which to save the collected examples (default: 600 Hz)')
parser.add_argument('-T', '--duration', type=float, default=6,
                    help='desired signal duration for each collected example (default: 6 s)')
parser.add_argument('-s', '--save_name', type=str, default='ccb_noise',
                    help='name of data save files')
args = parser.parse_args()

def read_audio_section(wav_path, start_time, end_time, sr):
    """
    Read a section of a WAV file.

    Parameters
    ----------
    wav_path : str
        path to WAV file
    start_time : float
        start time of section to read
    end_time : float
        end time of section to read
    sr : float
        sample frequency
    
    Returns
    -------
    array-like
        section of audio from WAV file
    """
    
    wav, _ = librosa.load(wav_path, offset=start_time, duration=(end_time - start_time), sr=sr)

    return wav

def collect_noise(n, fs_target_py, fs_target_mat, T_target):
    """
    Collect noise examples by randomly collecting 6-second examples from the data.

    Parameters
    ----------
    n : int
        number of samples to collect
    fs_target : float
        sampling frequency for resampling
    T_target : float
        length of signals to sample
    """

    # get deepest directories in provided in data root which contain the WAV files
    all_dirs = []
    max_depth = 0
    for path, subdirs, _ in os.walk(config['dataset']['ccb_data_directory']):
        for dir in subdirs:
            
            curr_path = os.path.join(path, dir)
            len_path = len(curr_path.split('/'))

            if len_path > max_depth:
                max_depth = len_path

            all_dirs.append((curr_path, len_path))
    wav_paths = [pair[0] for pair in all_dirs if pair[1] == max_depth]

    # to store noise signals
    X_mat = np.zeros((n, fs_target_mat*T_target))
    X_py = np.zeros((n, fs_target_py*T_target))

    files_map = dict()
    for i in tqdm(range(n)):
        
        # look for file with duration longer than T
        f_duration = 0
        while f_duration <= T_target:
            wav_path = str(np.random.choice(wav_paths))
            if wav_path not in files_map.keys():
                files_map[wav_path] = [f for f in os.listdir(wav_path) if os.path.isfile(os.path.join(wav_path, f)) and ".wav" in f]

            f_path = os.path.join(wav_path, np.random.choice(files_map[wav_path]))

            f_samplerate = librosa.get_samplerate(f_path)
            f_duration = librosa.get_duration(filename=f_path, sr=f_samplerate)

        # get random start time
        start_t = np.random.randint(0, np.floor(f_duration - T_target))
        end_t = start_t + T_target

        # read and mean-center signal
        wav = read_audio_section(f_path, start_t, end_t, f_samplerate)
        wav = wav - wav.mean()
        
        # downsample and save
        wav_mat = resample_poly(wav, fs_target_mat, f_samplerate)
        wav_py = resample_poly(wav, fs_target_py, f_samplerate)

        X_mat[i,:] = wav_mat
        X_py[i,:] = wav_py

    # save as MAT file
    mdict = {u'noise_from_data': X_mat.T, u'fs': float(fs_target_mat)}
    hdf5storage.savemat(os.path.join(config['dataset']['data_directory'], f"{args.save_name}.mat"), mdict, format="7.3")

    # mean-center and L2 norm for h5 noise
    X_py = X_py - X_py.mean(axis=1, keepdims=True)
    X_py = X_py / np.sqrt(np.sum(X_py ** 2, axis=1, keepdims=True))

    with h5py.File(os.path.join(config['dataset']['data_directory'], f"{args.save_name}.h5"), "w") as f:
        f.create_dataset('data', data=X_py, shape=X_py.shape, chunks=(1, X_py.shape[1]))
        f.create_dataset('fs', data=fs_target_py, shape=(1,)) 

if __name__ == "__main__":

    collect_noise(n=args.number, fs_target_py=args.sample_rate_python, fs_target_mat=args.sample_rate_matlab, T_target=args.duration)