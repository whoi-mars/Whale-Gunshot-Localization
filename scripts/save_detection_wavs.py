"""
Script to save spectrograms corresponding to a detected source.
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import soundfile as sf

from whale_gunshot_localization import config, PROJECT_ROOT_DIR
from whale_gunshot_localization.utils.experimental import load_wav
from whale_gunshot_localization.utils.transformations import to_spect

def get_wavs_by_id(df, Id):
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
        ID to detection wav files
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
        results.append(dict(zip(sensor_list, wav_list)))
    return results

if __name__ == "__main__":
    # create map of sensor number to sensor index
    sensor_to_id = dict()
    for i, id in enumerate(config['TOSSIT']['ids']):
        sensor_to_id[int(id)] = i+1

    ID = [67]
    # prepare directory to save images
    wav_dir = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", config['models']['model_dir'], "wavs")
    Path(wav_dir).mkdir(exist_ok=True, parents=True)

    # grab spectrograms
    df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "scripts", "results", config['models']['model_dir'], "multi_scan_results.csv"))
    results = get_wavs_by_id(df, ID)
    
    # save wavs
    for i, sig in enumerate(tqdm(results, disable=len(results) == 1, desc="saving wavs")):
        for sensor, wav in zip(sig.keys(),sig.values()):
            print(sensor)
            idx = sensor_to_id[sensor]
            sf.write(os.path.join(wav_dir,f"gunshot_{ID[i]}_{idx}.wav"),wav,config['signal']['fs'])
            