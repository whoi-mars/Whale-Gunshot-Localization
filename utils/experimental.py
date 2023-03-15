import os
import glob
import pickle
import warnings
import copy

import yaml
from bs4 import BeautifulSoup
import numpy as np

# load config file
with open("config.yaml", 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.Loader)

def get_matching_files(TOSSIT_dirs, minimum_file_duration=10, closeness_threshold=6):
    """
    Go through the XML files in the TOSSIT data and find which acoustic data files from each are "matching". That is, 
    find which acoustic data files are closes in time to each other according to 'WavFileHandler' tag with the
    'SamplingStartTimeLocal' attribute. The results are stored in a list of groups where each group is a list of
    tuples with each tuple containing (FILE_PATH, offset_from_files_in_first_tuple). Thus, the offset in the first
    typle in every list is 0.0 s.

    Parameters
    ----------
    TOSSIT_dirs : List
        list of directory names with each containing files associated with a single TOSSIT
    minimum_file_duration : float
        minimum duration of a file in the first TOSSIT directory required to attempt to find a match for it (in seconds)
    closeness_threshold : float
        minimum time difference between closest match found to keep it (in seconds)

    Returns
    -------
    matching_files: List[List[tuple]]
        matching results in the form -- [[(base_path, 0), (TOSSIT_2, delta_t_2), ..., (TOSSIT_n, delta_t_n)], ..., [...]]
    """    

    
    # to store results
    matching_files = []

    TOSSIT_paths = [os.path.join(config['dataset']['exp_data_directory'], d) for d in TOSSIT_dirs]

    # use first TOSSIT in provided list as base for file matching
    base_TOSSIT_path = TOSSIT_paths[0]
    base_TOSSIT_xmls = glob.glob(os.path.join(base_TOSSIT_path, "*.xml"))

    for base_xml in base_TOSSIT_xmls:
        
        # item to add to matching_files
        group = [(base_xml.split('log')[0]+"wav", 0.0)]

        with open(base_xml, 'r') as f:
            base_data = f.read()
        
        # get start/end times of base file
        base_data = BeautifulSoup(base_data, features="lxml")
        base_start_DT = np.datetime64(base_data.find_all("wavfilehandler", samplingstarttimelocal=True)[0]['samplingstarttimelocal'])
        base_end_DT = np.datetime64(base_data.find_all("wavfilehandler", samplingstoptimelocal=True)[0]['samplingstoptimelocal'])
        
        # if file is too short, skip it
        if base_end_DT - base_start_DT < np.timedelta64(minimum_file_duration, 'm'):
            continue

        # find xml file for every other TOSSIT which is closes in time
        for t in TOSSIT_paths[1:]:
            
            # get xml files for each TOSSIT
            TOSSIT_xmls = glob.glob(os.path.join(t, "*.xml"))
            DT_list = np.empty((len(TOSSIT_xmls,)), dtype='datetime64[s]')
            
            # iterate through xml files for TOSSIT to get
            # start times and find closest to current base_start_DT
            for i, xml in enumerate(TOSSIT_xmls):
                with open(xml, 'r') as f:
                    t_data = f.read()
                
                t_data = BeautifulSoup(t_data, features='lxml')
                DT_list[i] = np.datetime64(t_data.find_all("wavfilehandler", samplingstarttimelocal=True)[0]['samplingstarttimelocal'])

            # get index of closes file
            delta_T = np.abs(base_start_DT - DT_list)
            idx = np.argmin(delta_T)
            
            # ignore if above closeness threshold
            if delta_T[idx] > np.timedelta64(closeness_threshold, 'm'):
                warnings.warn(f'The closest match is between files {base_xml} and {xml}, but exceeds the closeness threshold of {closeness_threshold} minutes.' \
                            "Ignoring group.")
                continue

            # save to group    
            group.append((TOSSIT_xmls[idx].split('log')[0]+"wav", (base_start_DT - DT_list[idx]).astype(float)))
        
        # save group
        matching_files.append(group)

    return matching_files

if __name__ == "__main__":
    # if we already have the matching files stored, load them. if not, do the matching.
    if os.path.exists(os.path.join(config['dataset']['exp_data_directory'], "matching files.p")):
        with open(os.path.join(config['dataset']['exp_data_directory'], "matching files.p"), "rb") as f:
            matching_files = pickle.load(f)
    else:
        print("Matching corresponding files from each TOSSIT...")
        matching_files = get_matching_files(TOSSIT_dirs=["6468", "6470", "6471", "6474", "6476"])
        with open(os.path.join(config['dataset']['exp_data_directory'], "matching files.p"), "wb") as f:
            pickle.dump(copy.deepcopy(matching_files), f)