import os
import glob
import itertools
import warnings

from bs4 import BeautifulSoup
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.optimize import minimize
import gtsam
import pandas as pd

import hypernetx as hnx
import hypernetx.algorithms.hypergraph_modularity as hmod
import networkx as nx 

from whale_gunshot_localization.utils.transformations import to_spect
import whale_gunshot_localization.utils.math_tools as math_tools

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

def sort_wav_chronological(wav_files):
    """
    Sort wav file by start time and grab the start and end times
    for each of the files

    Parameters
    ----------
    wav_files : List[str]
        full path to each of the WAV files
    
    Returns
    -------
    wav_files : List[str]
        list of full paths to each of the WAV files sorted by start time
    start_times : List[np.datetime64]
        sorted list of start times
    end_times : List[np.datetime64]
        list of end times sorted by corresponding start times
    """

    start_times = []
    end_times = []

    for wf in wav_files:

        # get corresponding xml file
        xml_file = wf.split('.wav')[0] + ".log.xml"
        if not os.path.exists(xml_file):
            raise IOError(f"Corresponding XML file {xml_file} does not exist")

        # get start time
        with open(xml_file, 'r') as f:
            data = f.read()
        data = BeautifulSoup(data, features='lxml')
        start_time = np.datetime64(data.find_all("wavfilehandler", samplingstarttimelocal=True)[0]['samplingstarttimelocal'])
        end_time = np.datetime64(data.find_all("wavfilehandler", samplingstoptimelocal=True)[0]['samplingstoptimelocal'])
        start_times.append(start_time)
        end_times.append(end_time)

    # get sorted indices
    idx_sorted = np.argsort(start_times)

    start_times = np.asarray(start_times)
    end_times = np.asarray(end_times)
    
    return wav_files[idx_sorted], start_times[idx_sorted], end_times[idx_sorted]

def get_wav_day(wav_file, timestamp):
    """
    Get the day given a WAV file and a timestamp (in seconds).

    Parameters
    ----------
    wav_file : str
        full path to the WAV file
    timestamp : float
        timestamp (in seconds) for the provided WAV file

    Returns
    -------
    pd.datetime.date
        date of the timestamp within the provided WAV file
    """

    # get corresponding XML file
    xml_file = wav_file.split('.wav')[0] + ".log.xml"
    if not os.path.exists(xml_file):
        raise IOError(f"Corresponding XML file {xml_file} does not exist")
    
    # get start time
    with open(xml_file, 'r') as f:
        data = f.read()
    data = BeautifulSoup(data, features='lxml')
    start_time = np.datetime64(data.find_all("wavfilehandler", samplingstarttimelocal=True)[0]['samplingstarttimelocal'])

    # get time of timestamp
    curr_time = start_time + np.timedelta64(timestamp, 's')

    # extract day
    pd_curr_time = pd.to_datetime(curr_time)
    return pd_curr_time.date()

class Localizer:
    """
    Class to perform data association and range-based localization using 
    multiple range measurements from multiple sensors.
    
    ...
    
    Attributes
    ----------
    TOSSIT_locations : array-like of shape N X 2
        locations of the acoustic sensors or the form [y, x]
    measurements : List[array-like]
        each sublist contains range measurements associated with a particular sensor
    H : hypernetx.classes.hypergraph.Hypergraph
        hypergraph object
    k : int
        number of measurements to group when checking for measurement group consistency
    consistency_thresh : float
        threshold for distance from range to location to determine which measurements groups
        are self-consistent
    method_thresh : float in [0, 1]
        threshold to determine if approximation of H matrix from doi:10.1017/S0263574710000196
        is sufficient and determines whether to use closed-form or gradient-based localization approch
    prune : bool
        whether to prune measurement groups based on lack of range measurement intersection
    grid : bool
        whether to use grid method of localization rather than the iterative optimization approach
    rng : numpy.random._generator.Generator
        optional RNG object
    min_x : float
        minimum x coordinate used to generate simulated data for detection/range estimation network training.
        used to generate an initial guess for iterative optimization method.
    max_x : float
        maximum x coordinate used to generate simulated data for detection/range estimation network training.
        used to generate an initial guess for iterative optimization method.
    min_y : float
        minimum y coordinate used to generate simulated data for detection/range estimation network training.
        used to generate an initial guess for iterative optimization method.
    max_y : float
        maximum y coordinate used to generate simulated data for detection/range estimation network training.
        used to generate an initial guess for iterative optimization method.
    """
    
    def __init__(self, k, consistency_thresh=1000, method_thresh=0.95, prune=False, grid=False, rng=None):
        """
        Construct attributes
        
        Parameters
        ----------
        k : int
            number of measurements to group when checking for measurement group consistency
        consistency_thresh : float
            threshold for distance from range to location to determine which measurements groups
            are self-consistent
        method_thresh : float in [0, 1]
            threshold to determine if approximation of H matrix from doi:10.1017/S0263574710000196
            is sufficient and determines whether to use closed-form or gradient-based localization approch
        prune : bool
            whether to prune measurement groups based on lack of range measurement intersection
        grid : bool
            whether to use grid method of localization rather than the iterative optimization approach
        rng : numpy.random._generator.Generator
            optional RNG object
        """
        
        # load TOSSIT locations
        self.TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
        
        # initialize empty measurements and hypergraph
        self.measurements = None
        self.formatted_linear_idx = None
        self.H = None
        self._linear_measurements = None
        self._tuple_idx = None
        self._sensors = None
        self._ranges = None
        
        # uniformity constant for hypergraph
        if k > 3 and k <= self.TOSSIT_locations.shape[0]:
            self.k = k
        else:
            raise ValueError(f"k must be > 3 and < {self.TOSSIT_locations.shape[0]}.")
        
        # thresholds
        self.consistency_thresh = consistency_thresh
        self.method_thresh = method_thresh
        
        # prune sets of k measurements based on pairwise intersection check
        self.prune = prune
        
        # get bounds of considered grid
        self.min_x = config['scaling']['min_x']
        self.max_x = config['scaling']['max_x']
        self.min_y = config['scaling']['min_y']
        self.max_y = config['scaling']['max_y']

        # rng
        self.rng = rng if rng is not None else np.random

        if grid:
            self._make_LUT()
            self._localize = self._localize_grid
        else:
            self._localize = self._localize_opt_hybrid
    
    def _make_LUT(self):
        # precalculate TOSSIT distances for all possible gridded locations
        num_TOSSITs = self.TOSSIT_locations.shape[0]
        x = np.arange(self.min_x-10000, self.max_x+10000, 50)
        y = np.arange(self.min_y-10000, self.max_y+10000, 50)
        self.X, self.Y = np.meshgrid(x, y)
        self.LUT = np.zeros((num_TOSSITs, *self.X.shape))
        for t in range(num_TOSSITs):
            self.LUT[t,...] = np.sqrt(((self.TOSSIT_locations[t,0] - self.Y) ** 2) + ((self.TOSSIT_locations[t,1] - self.X) ** 2))
    
    def _localize_grid(self, ranges, sensors_idx):
        ranges = ranges[:,np.newaxis,np.newaxis]
        sq_err = ((self.LUT[sensors_idx,:] - ranges) ** 2).sum(axis=0)

        return np.min(sq_err), np.asarray([self.Y[sq_err == np.min(sq_err)], self.X[sq_err == np.min(sq_err)]]).squeeze()

    def _localize_opt(self, ranges, sensors_idx):
        """
        An iterative localization method based on minimizing the following cost function:
        
        (1 / N) * \sum_{i=1}^{N} (||p_{i} - p_{0}||_{2} - r_{i})^{2}.
        
        Parameters
        ----------
        ranges : array-like of shape M,
            list of range measurements from source to sensor
        sensors_idx : array-like of shape M
            list of sensor indices associated with the provided range measurements
            
        Returns
        -------
        res.fun : float
            cost function value
        res.x : array-like
            optimized location
        """
                
        # define objective function
        def obj(x):
            # calculate l2s between TOSSIT positions and predicted location
            l2 = np.sqrt((self.TOSSIT_locations[sensors_idx,0] - x[0]) ** 2 + (self.TOSSIT_locations[sensors_idx,1] - x[1]) ** 2)

            # return sum squared error between l2s and predicted ranges
            return (1 / len(ranges))*np.sum((l2.squeeze() - ranges) ** 2)

        # random initial guess
        x0 = [self.rng.uniform(self.min_y, self.max_y), self.rng.uniform(self.min_x, self.max_x)]
        
        # optimize!
        res = minimize(obj, x0, method='Nelder-Mead', options={'disp': False})
        
        return res.fun, res.x
    
    def _localize_opt_hybrid(self, ranges, sensors_idx):
        """
        Hybrid localization method which choses between slower iterative method
        and the closed form method presented in doi:10.1017/S0263574710000196.
        
        Parameters
        ----------
        ranges : array-like of shape M,
            list of range measurements from source to sensor
        sensors_idx : array-like of shape M,
            list of sensor indices associated with the provided range measurements
            
        Returns
        -------
        loc : array-like
            optimized location
        cost : float
            cost of the localization objective function
        """
                
        # get sensors and save how many
        p_i = self.TOSSIT_locations.T[:,sensors_idx]
        N = p_i.shape[1]

        # calculate a, B, c, H
        a = (p_i * (p_i * p_i).sum(axis=0, keepdims=True) \
             - (ranges**2 * p_i)).sum(axis=1, keepdims=True) / N

        eye = np.dstack(N*[np.eye(2)])
        B = ((-2*np.einsum('ij,kj->jik', p_i, p_i)).sum(axis=0) \
             + (-(np.expand_dims((p_i * p_i).sum(axis=0, keepdims=True), axis=0) * eye) \
             + np.expand_dims(ranges**2, axis=0) * eye).sum(axis=2)) / N

        c = p_i.sum(axis=1, keepdims=True) / N

        H = (-2 / N)*(np.einsum('ij,kj->jik', p_i, p_i).sum(axis=0)) + (2*c @ c.T)
        
        if np.linalg.matrix_rank(H) < 2:
            raise RuntimeError('H matrix not full rank.')
        else:
            # check if approximation holds to make H matrix.
            # if not, use gradient-based localization
            f = a + B @ c + 2*c @ c.T @ c
        
            q = -np.linalg.inv(H) @ f

            D = B + 2*c @ c.T + (c.T @ c) * np.eye(2)
            H_hat = D - (q.T @ q)*np.eye(2)
            
            dist = math_tools.matrix_similarity(H, H_hat)
            
            if dist < self.method_thresh:
                cost, loc = self._localize_opt(ranges, sensors_idx)
                return cost, loc

            # calculate cost for closed form method
            loc = (q + c).squeeze()
            l2 = np.sqrt((self.TOSSIT_locations[sensors_idx,0] - loc[0]) ** 2 + (self.TOSSIT_locations[sensors_idx,1] - loc[1]) ** 2)
            cost = (1 / len(ranges))*np.sum((l2.squeeze() - ranges) ** 2)
            return cost, loc 
        
    def set_measurements(self, measurements, adaptive=False, adaptive_max=5000, threshold_delta=500):
        """
        Create a hypergraph which represents groups of k-consistent measurements.
        
        Parameters
        ----------
        measurements : List[array-like]
            each sublist contains range measurements associated with a particular sensor
        adaptive : bool
            whether to increase the consistency threshold if no consistent measurements are found
        adaptive_max : float
            maximum consistency threshold to use before giving up when operating adaptively
        threshold_delta : float
            how much to increase the consistency threhold if no consistent groups are found when adaptive

        Returns
        -------
        bool
            whether associated data was found (True) or not (False)
        """
        
        # update measurements
        self.measurements = measurements

        ######################################################
        #               create formatted lists               #
        ######################################################

        # set of sensor indices with measurements
        non_empty_sensors = set()
        # list of all measurements traversed sensor-major
        self._linear_measurements = []
        # same shape as measurements but with linear indices
        self.formatted_linear_idx = [[] for _ in range(self.TOSSIT_locations.shape[0])]
        # measurement to linear index dict
        measurement_to_linear_idx = {}
        # (sensor_id, measurement_id) elements
        self._tuple_idx = []
        self._sensors = []
        self._ranges = []
        for i, m_list in enumerate(self.measurements):
            for j, m in enumerate(m_list):

                # keep track of sensors with measurements
                non_empty_sensors.add(i)

                # keep track of linear index
                linear_idx = len(self._linear_measurements)
                self.formatted_linear_idx[i].append(linear_idx)
                measurement_to_linear_idx[m] = linear_idx

                # keep linear list of measurements
                self._linear_measurements.append(m)

                self._tuple_idx.append((i, j))
                self._sensors.append(i)
                self._ranges.append(m)

        # convert linear_measurements to numpy array
        self._linear_measurements = np.asarray(self._linear_measurements)
        self._tuple_idx = np.asarray(self._tuple_idx)
        self._sensors = np.asarray(self._sensors)
        self._ranges = np.asarray(self._ranges)

        ######################################################
        #                  build hypergraph                  #
        ######################################################

        adaptive_thresh = self.consistency_thresh
        while True:
            # get all combinations of sensor indices
            sensor_combs = itertools.combinations(non_empty_sensors, self.k)

            # for each group of k sensors, get valid measurement combination
            # candidates based on trilateration errors
            scenes = {}
            edge_set_counter = 0
            for s_comb in sensor_combs:

                # get range measurment indices associated with specified sensors in s_comb
                ranges = [self.formatted_linear_idx[sensor_idx] for sensor_idx in s_comb]

                # get all combinations of measurement indices across the sensors as List[List]
                range_combos = map(list, itertools.product(*ranges))

                # determine which candidates are consistent
                for range_combo in range_combos:

                    range_subcombos = list(map(list,itertools.combinations(range_combo, self.k-1)))
                    s_subcombos = list(map(list, itertools.combinations(s_comb, self.k-1)))

                    append = True
                    for range_subcombo, s_subcombo in zip(range_subcombos, s_subcombos):

                        # prune groups of k measurements based if any of the pairs don't intersect
                        if self.prune and (not math_tools.check_intersection(self.TOSSIT_locations[s_subcombo,:], self._linear_measurements[range_subcombo])):
                            append = False
                            break

                        # get trilateration cost for candidate
                        _, loc = self._localize(self._linear_measurements[range_subcombo], s_subcombo)

                        r = (set(range_combo) - set(range_subcombo)).pop()
                        s = (set(s_comb) - set(s_subcombo)).pop()
                        if math_tools.point_circle_shortest_distance(self.TOSSIT_locations[s,:], self._linear_measurements[r], loc) > adaptive_thresh:
                            append = False
                            break

                    if append:
                        scenes[edge_set_counter] = range_combo
                        edge_set_counter += 1
            
            # save hypergraph
            if len(scenes):
                self.H = hnx.Hypergraph(scenes)
                return True
            elif adaptive and (not len(scenes)) and adaptive_thresh < adaptive_max:
                adaptive_thresh += threshold_delta
                continue
            else:
                return False

    
    def reset(self):
        """
        Reset object between different data
        """
        
        self.measurements = None
        self.formatted_linear_idx = None
        self.H = None
        self._linear_measurements = None
        self._tuple_idx = None
        self._sensors = None
        self._ranges = None

    def _last_step(self, associations):
        """
        Greedy algorithm to make sure each association has only one measurement from any given sensor.
        It finds the lowest cost member of each association among multiple measurements from a given sensor
        if there are any.

        Parameters
        ----------
        assocaitions : List[Set]
            Input associations
        
        Returns
        -------
        List[Set]
            Final associations
        """

        assert self.measurements, "measurements have not been set"

        for a_idx in range(len(associations)):
            for m in self.formatted_linear_idx:
                
                # set of measurements which must be disjoint
                mask = set(m)

                # intersect mask with associations to find
                # where more than one member is present
                intersection = mask.intersection(associations[a_idx])

                # figure out which member of intersection
                # is the best fit based on localization
                if len(intersection) > 1:
                    max_err = float('inf')
                    best_assoc = None
                    for i in itertools.combinations(intersection, len(intersection) - 1):

                        # take away all but one measurement from the sensor measurements
                        popped_a = associations[a_idx] - set(i)

                        # get sensor indices and ranges
                        sensors = self._sensors[list(popped_a)]
                        ranges = self._ranges[list(popped_a)]
                        
                        # calculate localization cost and update best
                        cost, _ = self._localize(ranges, sensors)
                        if cost < max_err:
                            max_err = cost
                            best_assoc = popped_a
                    
                    # update association
                    associations[a_idx] = best_assoc

        return [assoc for assoc in associations if len(assoc) >= 3]

    def associate_and_localize(self, method='clique'):
        """
        Using the hypergraph constructed in self.set_measurements, perform data association
        and localization.
        
        Parameters
        ----------
        method : string
            can be either 'clique' (doesn't really work right now) or 'partition'

        Returns
        -------
        associations : List[List] if method is 'clique, List[Set] if method is 'partition'
            resulting data associations where grouped numbers represent the linear measurement
            index, traversing self.measurements in sensor-major order
        locs : array-like of shape N X 2
            predicted locations of each association set in corresponding order
        """
        
        # make sure we've set measurements
        assert self.measurements, "measurements have not been set"
        
        if method == 'clique':
            associations = []
            for h in self.H.connected_component_subgraphs():
                for clique in nx.find_cliques_recursive(h):
                    if len(clique) >= 3:
                        associations.append(clique)
        elif method == 'partition':
            HG = hmod.precompute_attributes(self.H)
            associations = hmod.kumar(HG)
            associations = self._last_step(associations)
        else:
            raise ValueError("Method must be either 'clique' or 'partition'")

        locs = []
        for a in associations:
            a = list(a)
            sensors = [self._tuple_idx[idx][0] for idx in a]
            ranges = self._linear_measurements[a]
            _, loc = self._localize(ranges, sensors)
            locs.append(loc)
        locs = np.asarray(locs)
        
        return associations, locs

def l2_standardize(examples, mu_list, std_list):
    """
    Mean-center, L2 normalizer, convert to spectrogram, and standardize a
    batch of inputs.

    Parameters
    ----------
    examples : array-like, (# examples, samples / example)
        raw collated examples from experimental data
    mu_list : array-like
        mean for each spectrogram row across training set
    std_list : array-like
        std for each spectrogram row across training set
    
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
    spect_examples = (spect_examples - mu_list) / std_list

    # return as PyTorch Tensor
    return torch.from_numpy(spect_examples).float()


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
    fs : float
        sampling frequency
    T : float
        signal duration
    preproessor : function
        function to preprocess a batch of signals in a way appropriate for the input model.
        should return a batch of PyTorch Tensors.
    device : torch.device
        device to run the model with
    overlap_fraction : float
        percentage of sample length to overlap
    sensitivity : int
        number of adjacent detections to certify an actual detection
    """

    def __init__(self, model, data_params, preprocessor, device='cpu', overlap_fraction=0.75, sensitivity=1):
        """
        Prepare model and constants.

        Parametes
        ---------
        model : torch.nn.Module
            model which performs detection and range estimation
        data_params : dict
            dictionary containing the following key-value pairs:
            mu_list : array-like
                mean for each spectrogram row across training set
            std_list : array-like
                std for each spectrogram row across training set
            fs : float
                sampling frequency
            T : float
                signal duration
        preproessor : function
            function to preprocess a batch of signals in a way appropriate for the input model.
            should take in 'examples', 'mu_list', and 'std_list' and return a batch of PyTorch Tensors.
        device : torch.device or str
            device to run the model with
        overlap_fraction : float
            percentage of sample length to overlap
        sensitivity : int
            number of adjacent detections to certify an actual detection
        """

        # model and device to use in analysis
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

        # method to preprocess a batch of data
        self.preprocessor = preprocessor

        # mu/std for standardizing data
        self.mu_list = np.expand_dims(data_params['mu_list'], axis=-1)
        self.std_list = np.expand_dims(data_params['std_list'], axis=-1)

        # size of window for model
        self.fs = data_params['fs']
        self.T = data_params['T']
        self.size = self.fs * self.T

        # overlap fraction between windows for model analysis
        self.overlap_fraction = overlap_fraction

        # minimum peak of detection plateau to certify a detection
        self.sensitivity = sensitivity
        assert self.sensitivity in (np.arange((1 / (1 - self.overlap_fraction))) + 1), f"with overlap_fraction = {self.overlap_fraction}, sensitivity must be in [1, {int(1 / (1 - self.overlap_fraction))}]"

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

    def _filter_model_outputs(self, examples, outputs):
        """
        Filter out unique call detections from the analysis of
        a batch of overlapping snippets of experimental data. 
        """
        
        # get boolean vector of detections
        detection_vec = torch.argmax(outputs[:,1:], dim=1)

        # get indices of unique detections. add padding to detect leading peaks
        detection_vec = np.pad(detection_vec, 1)
        detection_idx, _ = find_peaks(detection_vec, plateau_size=self.sensitivity)
        detection_idx = detection_idx - 1

        # ensure detections are separated by T
        while True:

            # detect repeat detections. add padding for leading detections
            diff_detection = (np.diff(detection_idx) * (1 - self.overlap_fraction) * self.T) < self.T
            diff_detection = np.pad(diff_detection, 1)
            peaks, props = find_peaks(diff_detection, plateau_size=1)

            if len(props['plateau_sizes']) == 0:
                break

            to_delete = []
            for w, l in zip(props['plateau_sizes'], props['left_edges']):
                to_delete.append(l + 1 if w > 1 else l)
            detection_idx = np.delete(detection_idx, to_delete)
        
        # filter examples and outputs
        examples = examples[detection_idx,:]
        outputs = outputs[detection_idx,:]
        timestamps = detection_idx * (1 - self.overlap_fraction) * self.T

        # get detected ranges
        return examples, (outputs[:,0]*config['scaling']['max_r']).numpy(), timestamps

    def process_clips(self, clips):

        # get batch of windows
        collated_batch = self._collate_samples(clips)

        # preprocessing
        preprocessed_batch = self.preprocessor(collated_batch, self.mu_list, self.std_list).to(self.device) # self._preprocess_batch(collated_batch).to(self.device)

        # run through model
        with torch.set_grad_enabled(False):
            outputs = [self.model(preprocessed_batch[:,i,...]).to('cpu') for i in range(preprocessed_batch.shape[1])]

        # filter unique detections and ranges
        return [self._filter_model_outputs(preprocessed_batch[:,i,...].to('cpu'), outputs[i]) for i in range(preprocessed_batch.shape[1])]
