import os
import yaml
import pickle
import copy

import librosa

import utils.experimental as experimental

# load config file
with open("config.yaml", 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.Loader)

# constants
MINUTES_TO_SECONDS = 60

# configuration
# -------------
# half the width of the window to look for matching signals in non-anchor TOSSITs
discovery_window = 60 # [sec]
# half chunk of data from anchor to load at a time
anchor_chunk_size = 10 # [min]

# if we already have the matching files stored, load them. if not, do the matching.
if os.path.exists(os.path.join(config['dataset']['ccb_data_directory'], "matching files.p")):
    print("Found matched TOSSIT files...")
    with open(os.path.join(config['dataset']['ccb_data_directory'], "matching files.p"), "rb") as f:
        from collections import Counter
        matching_files = pickle.load(f)
else:
    print("Matching corresponding files from each TOSSIT...")
    matching_files = experimental.get_matching_files(TOSSIT_dirs=["6468", "6470", "6471", "6474", "6476"])
    with open(os.path.join(config['dataset']['ccb_data_directory'], "matching files.p"), "wb") as f:
        pickle.dump(copy.deepcopy(matching_files), f)

# pick a file collection 
collection = matching_files[0]
anchor_path = collection[0][0]
anchor_fs = librosa.get_samplerate(anchor_path)
anchor_duration = librosa.get_duration(filename=anchor_path)

# get starting point
offsets = [pair[1] for pair in collection]
max_abs_offsets = max(offsets, key=lambda x: abs(x))
if max_abs_offsets <= 0 and abs(max_abs_offsets) < discovery_window:
    anchor_pointer = discovery_window
elif max_abs_offsets > 0:
    anchor_pointer = max_abs_offsets + discovery_window
else:
    anchor_pointer = 0

while (anchor_duration - anchor_pointer) > anchor_chunk_size*MINUTES_TO_SECONDS:
    
    # load a chunk and move pointer
    anchor_chunk, _ = librosa.load(anchor_path, sr=anchor_fs, offset=anchor_pointer, duration=anchor_chunk_size*MINUTES_TO_SECONDS)
    anchor_pointer += anchor_chunk_size*MINUTES_TO_SECONDS






