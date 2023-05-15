import os
import glob
import warnings
import copy

import yaml
from bs4 import BeautifulSoup
import numpy as np
import torch
from scipy.signal import find_peaks

from whale_gunshot_localization.utils.transformations import to_spect

# load config file
from whale_gunshot_localization import config

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

    TOSSIT_paths = [os.path.join(config['dataset']['ccb_data_directory'], d) for d in TOSSIT_dirs]

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

        # flag for if a match can't be found among the other TOSSITs
        missing_match = False

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
                
                # trigger flag
                missing_match = True

                # stop matching process for this TOSSIT
                break

            # save to group    
            group.append((TOSSIT_xmls[idx].split('log')[0]+"wav", (base_start_DT - DT_list[idx]).astype(float)))
        
        # save group
        if not missing_match:
            matching_files.append(group)

    return matching_files

class ClipAnalyzer:

    """
    Class which applies trained model to experimental data to get detections
    and range measurements.

    ...

    Attributes
    ----------
    model : torch.nn.Module
        model which performs detection and range estimation
    mu_list : array-like
        mean for each spectrogram row across training set
    std_list : array-like
        std for each spectrogram row across training set
    size : int
        size of the overlapping chunks
    device : torch.device
        device to run the model with
    overlap_fraction : float
        percentage of sample length to overlap
    """

    def __init__(self, model, mu_list, std_list, size, device, overlap_fraction=0.75):
        """
        Prepare model and constants.

        Parametes
        ---------
        model : torch.nn.Module
            model which performs detection and range estimation
        mu_list : array-like
            mean for each spectrogram row across training set
        std_list : array-like
            std for each spectrogram row across training set
        size : int
            size of the overlapping chunks
        device : torch.device
            device to run the model with
        overlap_fraction : float
            percentage of sample length to overlap
        """

        # model and device to use in analysis
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # mu/std for standardizing data
        self.mu_list = np.expand_dims(mu_list, axis=-1)
        self.std_list = np.expand_dims(std_list, axis=-1)

        # size of window for model
        self.size = size

        # overlap fraction between windows for model analysis
        self.overlap_fraction = overlap_fraction

    def _collate_samples(self, signal, n=float('inf')):
        """
        Collate short snippits from an audio signal with 50% overlap.

        Parameters
        ----------
        signal : array-like, (1, N) or (N,)
            long signal to split into smaller overlapping chunks
        n : int
            number of overlapping chunks to sequentially generate from 'signal'
        """

        assert self.overlap_fraction < 1 and self.overlap_fraction > 0, "overlap_fraction must be in (0,1)."

        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)

        # we want to overlap the signal between samples
        overlap = int(self.size*(1 - self.overlap_fraction))
        
        # get samples which meet the size requirement
        collated = np.asarray([signal[:,i:i+self.size] for sample, i in enumerate(range(0, signal.shape[1], overlap), start=1) if signal.shape[1] - i >= self.size and sample <= n])

        return collated

    def _preproess_batch(self, examples):
        """
        Apply all necessary preprocessing transforms.

        Parameters
        ----------
        examples : array-like, (# examples, samples / example)
            raw collated examples from experimental data
        
        Returns
        -------
        torch.Tensor, (# examples, # frequency bins, # time bins)
            preprocessed collabed examples from experimental data
        """

        # mean-center
        examples = examples - examples.mean(axis=2, keepdims=True)

        # l2 norm
        examples = examples / np.sqrt(np.sum(examples ** 2, axis=2, keepdims=True))

        # convert to spectrograms
        spect_examples = to_spect(examples).squeeze(axis=2)

        # standardize
        spect_examples = (spect_examples - self.mu_list) / self.std_list

        # return as PyTorch Tensor
        return torch.from_numpy(spect_examples).float()

    def _filter_model_outputs(self, examples, outputs):
        """
        Filter out unique call detections from the analysis of
        a batch of overlapping snippets of experimental data. 
        """
        
        # get boolean vector of detections
        detection_vec = torch.argmax(outputs[:,1:], dim=1)

        # get indices of unique detections and filter examples and outputs
        detection_idx, _ = find_peaks(detection_vec)
        examples = examples[detection_idx,:]
        outputs = outputs[detection_idx,:]

        # get detected ranges
        return examples, outputs[:,0]*config['scaling']['max_r']

    def process_clips(self, clips):

        # get batch of windows
        collated_batch = self._collate_samples(clips)

        # preprocessing
        preprocessed_batch = self._preproess_batch(collated_batch).to(self.device)

        # run through model
        with torch.set_grad_enabled(False):
            outputs = [self.model(preprocessed_batch[:,i,...]).to('cpu') for i in range(preprocessed_batch.shape[1])]

        # filter unique detections and ranges
        return [self._filter_model_outputs(preprocessed_batch[:,i,...].to('cpu'), outputs[i]) for i in range(preprocessed_batch.shape[1])]