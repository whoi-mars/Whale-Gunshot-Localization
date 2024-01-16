from scipy.io import wavfile
from scipy.signal import decimate, stft
import matplotlib.pyplot as plt
from  matplotlib.widgets import Button
import random
import numpy as np
import argparse
import csv
import os

from whale_gunshot_localization import config, PROJECT_ROOT_DIR

def get_undecimated_indices(start_idx_dec, fs_dec, fs, T):

    """
    Translates the start and end indices of a decimated signal back to what they
    would be in the original undecimated file.

    Parameters
    ----------
    start_idx_dec: int, start index of the decimated sitnal
    fs_dec: float, sampling frequency of the decimated signal
    fs: int, sampling frequency of the undecimated signal
    T: Duration of the signal

    Returns
    -------
    start_idx: int, start index of undecimated signal
    end_idx: int, end index of undecimated signal
    """

    # Get time in call of start index
    t = start_idx_dec / fs_dec

    # Get index of start time in undecimated signal
    start_idx = t*fs

    # Number of samples for desired time window
    n_samples = np.ceil(T*fs)

    # Make sure indxs are ints
    start_idx = int(start_idx)
    end_idx = int(start_idx + n_samples)

    return start_idx, end_idx

def wav_rejection_sample(wav_paths, csv_path, T, fmax):

    """
    Plots a grid of randomly selected signals from provided wav file directories
    of a specified length. The user can click on those that they do not want to be
    saved (hence rejection sampling), and then hit the 'Save' button to save the file name,
    start index in the undecimated file, and end index in the undecimated file to a CSV.
    Pushing the 'Exit' button will terminate the program and nothing further will be saved.

    Example
    -------
    python -m range_finder.scripts.wav_rejection_sample --save noise_collect

    Inputs
    ------
    wav_paths: array[str], path (paths) to a directory (directories) containing the wav files to sample from
    csv_path: str, path of the CSV file to save results to
    fmax: float, The maximum frequency to show in the displayed spectrograms 
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
        
        # Choose path to wav file directory randomly
        wav_path = random.choice(wav_paths)

        # Collect wave files from 
        wav_files = [f for f in os.listdir(wav_path) if os.path.isfile(os.path.join(wav_path, f)) and ".wav" in f]

        # Choose random wav file and load
        wav_file = random.choice(wav_files)
        print(os.path.join(wav_path, wav_file))
        fs, s = wavfile.read(os.path.join(wav_path, wav_file))

        # Decimate signal appropriatly
        dec_factor = round(fs / (2*fmax))
        s_dec = decimate(s, dec_factor)
        fs_dec = fs / dec_factor

        # Number of samples per specified time window
        n_samples = round(T*fs_dec)

        # Make numpy array to store files names and indices
        rows = np.zeros((square ** 2, 3), dtype=object)

        # Save axes
        ax_list = list()

        # Choose a random file and plot a grid of samples
        ix = 1
        fig = plt.figure('Rejection Sampler', figsize=(32,32))
        plt.clf()
        for _ in range(square):
            for _ in range(square):
                
                # Choose random sample from chosen file and store indices
                dec_start_idx = random.randint(0, len(s_dec) - n_samples)
                start_idx, end_idx = get_undecimated_indices(dec_start_idx, fs_dec, fs, T)
                rows[ix-1,:] = [os.path.join(wav_path, wav_file), start_idx, end_idx]

                # Calcualte spectrogram
                s_curr = s_dec[dec_start_idx:dec_start_idx+n_samples]
                [f,t,Zxx] = stft(x=s_curr, fs=fs_dec, nperseg=31, noverlap=20, nfft=500)
                log_spec = np.flipud(10*np.log10(np.abs(Zxx)**2))
                print(log_spec.shape)

                ax = fig.add_subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(log_spec)
                ax_list.append(ax)
                ix += 1

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
                writer.writerow(["file", "start_ind", "end_ind"])
                writer.writerows(rows)
        else:
            with open(csv_path, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)

        # Clear unwanted axes
        wanted_ax_set = set(range(square ** 2))

if __name__ == "__main__":

    all_dirs = list()
    max_depth = 0
    for path, subdirs, _ in os.walk(config['ccb_data_directory']):
        for dir in subdirs:
            
            curr_path = os.path.join(path, dir)
            len_path = len(curr_path.split('/'))

            if len_path > max_depth:
                max_depth = len_path

            all_dirs.append((curr_path, len_path))

    
    wav_paths = [pair[0] for pair in all_dirs if pair[1] == max_depth]

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", type=str, help="Name of CSV log_file")

    # Decode arguments
    args = parser.parse_args()
    csv_name = 'results' if args.save is None else args.save

    wav_rejection_sample(
        wav_paths, 
        os.path.join('/tf/workspace/range_finder/data/csv_files/', '{}.csv'.format(csv_name)), 
        config['signal']['T'], 
        config['signal']['fs']
    )