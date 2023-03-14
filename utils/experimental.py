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

class ExperimentalData():

    def __init__(self, TOSSIT_files, minimum_file_duration=10, closeness_threshold=6):
        
        self.TOSSIT_paths = [os.path.join(config['dataset']['exp_data_directory'], d) for d in TOSSIT_files]
        self.minimum_file_duration = minimum_file_duration
        self.closeness_threshold = closeness_threshold

        # if we already have the matching files stored, load them. if not, do the matching.
        if os.path.exists(os.path.join(config['dataset']['exp_data_directory'], "matching files.p")):
            with open(os.path.join(config['dataset']['exp_data_directory'], "matching files.p"), "rb") as f:
                self.matching_files = pickle.load(f)
        else:
            print("Matching corresponding files from each TOSSIT...")
            self.matching_files = self._get_matching_files()
            with open(os.path.join(config['dataset']['exp_data_directory'], "matching files.p"), "wb") as f:
                pickle.dump(copy.deepcopy(self.matching_files), f)

    def _get_matching_files(self):
        
        # [[(base_path, 0), (TOSSIT_2, delta_t_2), ..., (TOSSIT_n, delta_t_n)], ..., [...]]
        matching_files = []

        # use first TOSSIT in provided list as base for file matching
        base_TOSSIT_path = self.TOSSIT_paths[0]
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
            if base_end_DT - base_start_DT < np.timedelta64(self.minimum_file_duration, 'm'):
                continue

            # find xml file for every other TOSSIT which is closes in time
            for t in self.TOSSIT_paths[1:]:
                
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
                if delta_T[idx] > np.timedelta64(self.closeness_threshold, 'm'):
                    warnings.warn(f'The closest match is between files {base_xml} and {xml}, but exceeds the closeness threshold of {self.closeness_threshold} minutes.' \
                                "Ignoring group.")
                    continue

                # save to group    
                group.append((TOSSIT_xmls[idx].split('log')[0]+"wav", (base_start_DT - DT_list[idx]).astype(float)))
            
            # save group
            matching_files.append(group)

        return matching_files

    def _spectrogram_corr(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

if __name__ == "__main__":
    ED = ExperimentalData(TOSSIT_files=["6468", "6470", "6471", "6474", "6476"])