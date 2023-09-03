"""
Script to save spectrograms corresponding to a detected source.
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from whale_gunshot_localization import config, PROJECT_ROOT_DIR
from whale_gunshot_localization.utils.experimental import load_wav
from whale_gunshot_localization.utils.transformations import to_spect

def convert_range(old_range, new_range, old_value):
    """
    Linearly map a number from one range to another range.

    Parameters
    ----------
    old_range : array-like, shape: 1 X 2
        minimum and maximum of the old number range
    new_range : array-like, shape: 1 X 2
        minimum and maximum of the new number range
    old_value : float
        value to map onto new range
    
    Returns
    -------
    : float
        value mappend onto new range
    """

    old_diff = (old_range[1] - old_range[0])  
    new_diff = (new_range[1] - new_range[0])  
    return (((old_value - old_range[0]) * new_diff) / old_diff) + new_range[0]

def get_spects_by_id(df, Id):
    """
    Get spectrograms and corresponding sensors.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing the source detections
    Id : int or List[int]
        source ID or list of source IDs to save spectrograms for
    
    Returns
    -------
    : List[Dict[int, array-like]]
        list of dictionaries, each corresponding to a source and mapping from sensor
        ID to detection spectrogram
    """

    if isinstance(Id, int):
        Id = [Id]
    elif isinstance(Id, list):
        assert len(Id) == len([i for i in Id if isinstance(i, int)])
    
    results = []
    for i in tqdm(Id, disable=len(Id) == 1, desc="calculating spectrograms"):
        # get dataframe corresponding to ID
        df_id = df[df['id'] == i]

        wav_list = []
        sensor_list = []
        for _, row in df_id.iterrows():
            sensor_list.append(row['sensor'])
            wav_list.append(load_wav(row['file_name'], row['timestamp'], 6))

        wav_list = np.asarray(wav_list)
        spect_list = to_spect(wav_list).squeeze()
        results.append(dict(zip(sensor_list, spect_list)))
    return results

if __name__ == "__main__":

    # prepare directory to save images
    fig_dir = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", config['models']['model_dir'], "detection_spects")
    Path(fig_dir).mkdir(exist_ok=True, parents=True)

    # grab spectrograms
    df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "scripts", "results", config['models']['model_dir'], "multi_scan_results.csv"))
    results = get_spects_by_id(df, np.arange(1, df['id'].max()))
    
    # make and save figures
    for ID, l in enumerate(tqdm(results, disable=len(results) == 1, desc="saving figures"), start=1):
        fig, axs = plt.subplots(len(l.keys()), 1, sharex=True)
        num_spects = len(l)
        for i, (sensor, spect) in enumerate(l.items()):
            # plot spectrogram
            axs[i].imshow(spect)

            # set x ticks
            x_tick_labels = np.arange(0, config['signal']['T'] + 1, 1)
            old_range_x = [0, config['signal']['T']]
            new_range_x = [0, spect.shape[1]]
            axs[i].set_xticks([convert_range(old_range_x, new_range_x, val) for val in x_tick_labels])
            axs[i].set_xticklabels([int(val) if val.is_integer() else val for val in x_tick_labels])
            
            # set y ticks
            y_tick_labels = np.arange(0, config['signal']['fs'] / 2 + 1, 100)
            old_range_y = [0, config['signal']['fs'] / 2]
            new_range_y = [0, spect.shape[0]]
            axs[i].set_yticks([convert_range(old_range_y, new_range_y, val) for val in y_tick_labels])
            axs[i].set_yticklabels(reversed([int(val) if val.is_integer() else val for val in y_tick_labels]))
            
            axs[i].set_title(str(sensor))
            axs[i].set_ylabel("Frequency [Hz]")
            if i+1 == num_spects:
                axs[i].set_xlabel("Time [s]")
        
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'{ID}.png'))
        plt.close(fig)