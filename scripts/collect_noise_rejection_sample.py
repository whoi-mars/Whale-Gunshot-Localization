from scipy.signal import stft, resample_poly
import matplotlib.pyplot as plt
from  matplotlib.widgets import Button
import random
import numpy as np
import argparse
import csv
from pathlib import Path
import librosa
import hdf5storage
import h5py
from tqdm import tqdm
import os

from whale_gunshot_localization import config, PROJECT_ROOT_DIR

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, help="Name of CSV log_file")
parser.add_argument("--T", type=str, default=10, help="Time window to save")
parser.add_argument("-s", "--save", action='store_true', help="Save collected data")
parser.add_argument('-fsmat', '--sample_rate_matlab', type=int, default=12000,
                    help='desired sample rate for MATLAB data simulation at which to save the collected examples (default: 12000 Hz)')
parser.add_argument('-fspy', '--sample_rate_python', type=int, default=600,
                    help='desired sample rate for NN training at which to save the collected examples (default: 600 Hz)')
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

def wav_rejection_sample(wav_paths, csv_path, T):

    """
    Plots a grid of randomly selected signals from provided wav file directories
    of a specified length. The user can click on those that they do not want to be
    saved (hence rejection sampling), and then hit the 'Save' button to save the file name,
    start index in the undecimated file, and end index in the undecimated file to a CSV.
    Pushing the 'Exit' button will terminate the program and nothing further will be saved.

    Inputs
    ------
    wav_paths: array[str] 
        path (paths) to a directory (directories) containing the wav files to sample from
    csv_path: str
        path of the CSV file to save results to
    T: float
        signal duration to save
    """

    # Size of one side of plot grid
    square = 3

    # On click event handler and global undwanted axes list
    wanted_ax_set = set(range(square ** 2))
    def on_plot_click(event, ax_list, wanted_ax_list):
        for i in ax_list:
            if event.inaxes == i and ax_list.index(i) in wanted_ax_list:
                wanted_ax_list.remove(ax_list.index(i))

    while True:
        
        # Choose random wav file and load
        selected_files = []
        selected_srs = []
        selected_durations = []
        for _ in range(square ** 2):
            wav_path = random.choice(wav_paths)
            wav_files = [f for f in os.listdir(wav_path) if os.path.isfile(os.path.join(wav_path, f)) and ".wav" in f]
            selected_files.append(os.path.join(wav_path, random.choice(wav_files)))
            selected_srs.append(librosa.get_samplerate(selected_files[-1]))
            selected_durations.append(np.floor(librosa.get_duration(filename=selected_files[-1], sr=selected_srs[-1])))

        # Make numpy array to store files names and indices
        rows = np.zeros((square ** 2, 4), dtype=object)

        # Save axes
        ax_list = list()

        # Choose a random file and plot a grid of samples
        ix = 1
        counter = 0
        fig = plt.figure('Rejection Sampler', figsize=(32,32))
        plt.clf()
        for _ in range(square):
            for _ in range(square):
                
                # Choose random sample from chosen file and store indices
                start_t = random.randint(0, selected_srs[counter] - T)
                end_t = start_t + T
                rows[ix-1,:] = [selected_files[counter], start_t, end_t, selected_srs[counter]]

                # Calcualte spectrogram
                s_curr = read_audio_section(selected_files[counter], start_t, end_t, selected_srs[counter])
                s_curr = s_curr - s_curr.mean()
                if config['signal']['fs'] != selected_srs[counter]:
                    s_curr = resample_poly(s_curr, config['signal']['fs'], selected_srs[counter])
                    s_curr = s_curr - s_curr.mean()

                [f,t,Zxx] = stft(x=s_curr, fs=config['signal']['fs'], nperseg=config['stft']['nperseg'], noverlap=config['stft']['noverlap'], nfft=config['stft']['nfft'])
                log_spec = np.flipud(10*np.log10(np.abs(Zxx)**2))
                print(log_spec.shape)

                ax = fig.add_subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(log_spec)
                ax_list.append(ax)
                ix += 1
                counter += 1

        # Set up selection of figures to not save        
        fig.canvas.mpl_connect("button_press_event", lambda event: on_plot_click(event, ax_list, wanted_ax_set))
        
        # Continue and Exit buttons
        axsave = plt.axes([0.7, 0.05, 0.1, 0.075])
        axexit = plt.axes([0.81, 0.05, 0.1, 0.075])
        bsave = Button(axsave, 'Save')
        bsave.on_clicked(lambda event: plt.close(fig))
        bsave.label.set_fontsize(28)
        bexit = Button(axexit, 'Exit')
        bexit.on_clicked(lambda event: exit(0))
        bexit.label.set_fontsize(28)
        plt.show()
        
        # Filter rows to keep what we want
        rows = rows[list(wanted_ax_set),:]
        print("Saving {} signals".format(rows.shape[0]))

        if not os.path.exists(csv_path):
            with open(csv_path, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["file", "start_t", "end_t", "fs"])
                writer.writerows(rows)
        else:
            with open(csv_path, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)

        # Clear unwanted axes
        wanted_ax_set = set(range(square ** 2))

def save_data(csv_path, T):
    """
    Save selected data from CSV

    Inputs
    ------
    csv_path : str
        path to CSV with audio segments to save
    T : float
        duration of data to be saved
    """
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        reader = list(reader)
        n = len(reader)

        # to store noise signals
        X_mat = np.zeros((n, T*args.sample_rate_matlab))
        X_py = np.zeros((n, T*args.sample_rate_python))

        for i, row in enumerate(tqdm(reader)):
            wav = read_audio_section(row['file'], int(row['start_t']), int(row['end_t']), int(row['fs']))
            wav = wav - wav.mean()
            wav_mat = resample_poly(wav, args.sample_rate_matlab, int(row['fs'])) if args.sample_rate_matlab != row['fs'] else wav
            wav_py = resample_poly(wav, args.sample_rate_python, int(row['fs'])) if args.sample_rate_python != row['fs'] else wav
            X_mat[i,:] = wav_mat
            X_py[i,:] = wav_py

    # save as MAT file
    mdict = {u'noise_from_data': X_mat.T, u'fs': float(args.sample_rate_matlab)}
    hdf5storage.savemat(os.path.join(config['dataset']['data_directory'], f"{args.file}.mat"), mdict, format="7.3")

    # mean-center and L2 norm for h5 noise
    X_py = X_py - X_py.mean(axis=1, keepdims=True)
    X_py = X_py / np.sqrt(np.sum(X_py ** 2, axis=1, keepdims=True))

    with h5py.File(os.path.join(config['dataset']['data_directory'], f"{args.file}.h5"), "w") as f:
        f.create_dataset('data', data=X_py, shape=X_py.shape, chunks=(1, X_py.shape[1]))
        f.create_dataset('fs', data=args.sample_rate_matlab, shape=(1,)) 
                
if __name__ == "__main__":

    # file information
    noise_dir = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", "noise")
    Path(noise_dir).mkdir(exist_ok=True, parents=True)
    csv_name = 'results' if args.file is None else args.file

    if args.save:
        save_data(os.path.join(noise_dir, '{}.csv'.format(csv_name)), args.T)
    else:
        all_dirs = list()
        max_depth = 0
        for path, subdirs, _ in os.walk('/media/markgoldwater/Extreme SSD/CCB_2023'):
            for dir in subdirs:
                
                curr_path = os.path.join(path, dir)
                len_path = len(curr_path.split('/'))

                if len_path > max_depth:
                    max_depth = len_path

                all_dirs.append((curr_path, len_path))

        wav_paths = [pair[0] for pair in all_dirs if pair[1] == max_depth]

        wav_rejection_sample(
            wav_paths, 
            os.path.join(noise_dir, '{}.csv'.format(csv_name)), 
            args.T,
        )