import yaml

import torch
from torch.utils import data
import h5py
import numpy as np

from whale_gunshot_localization.utils.transformations import to_spect, random_wrap
from whale_gunshot_localization.datasets.samplers import H5BatchSampler, UniformGridH5BatchSampler
from whale_gunshot_localization import config

class SimDataRangeClassify(data.Dataset):
    """
    Dataset class for simulated data classification and ranging tasks. This dataloader is stochastic in that it, with 50-50 chance, chooses to 
    load noise or the signal. If it loads the signal it randomly chooses a TOSSIT to load the signal from. If it loads noise, the signal is
    randomly rotated.

    ...

    Attributes
    ----------
    data : HDF5 file
        pointer to (simulated) data-containing HDF5 file
    labels : HDF5 file
        pointer to (simulated) data labels HDF5 file
    range_labels : array-like
        the range of the source from each TOSSIT
    noise : HDF5 file
        pointer to (experimental) noise signals in HDF5 file
    num_noises : int
        the total number of experimental noise signals in 'noise'
    noise_inds : array-like
        array of indices for 'noise' which correspond to 'split' 
    size : tuple
        spectrogram shape
    num_TOSSITs : int
        number of TOSSITs used
    max_r : float
        maximum range in the dataset for scaling
    transform : PyTorch Compose object
        desired data transformations
    squeeze : bool
        whether or not to eliminate the singleton channel dimension
    track_TOSSIT : bool
        whether or not we use deterministic TOSSIT selection for each data example
    TOSSIT_LUT : array-like
        map from data index to TOSSIT to take data from 
    """

    def __init__(self, split, transform=None, squeeze=False, sensors=None):
        """
        Construct attributes and grab reference to data file

        Parameters
        ----------
        split : string
            which split to draw data from
        transform : PyTorch Compose object
            desired data transformations
        squeeze : bool
            whether or not to eliminate the singleton channel dimension 
        """

        super(SimDataRangeClassify, self).__init__()

        # make sure split is valid
        assert split in ['train', 'test', 'val'], "'split' must be one of 'train', 'test', 'val'"

        # get data file
        self.data, self.labels, self.range_labels, self.noise = self._load_h5()

        # get number of noise examples
        self.num_noises = self.noise.shape[0]
        self.noise_inds = np.load(config['dataset']['data_directory'] + f'/{split}_noise_indices.npy', allow_pickle=True)

        # for scaling location labels
        self.max_r = config['scaling']['max_r']

        # get imsize
        self.sensors = np.arange(self.data.shape[1]) if sensors is None else sensors
        self.size = to_spect(self.data[0]).shape[2:]

        self.transform = transform
        self.squeeze = squeeze

    def __getitem__(self, idx):

        # get data or noise randomly
        t_ind = 0 # np.random.choice(self.sensors)
        if np.random.choice([0, 1]):
            inputs = self.data[idx,[t_ind]]
            target_c = self._from_numpy(np.asarray([1]))
            target_r = self._from_numpy(self.range_labels[[idx],t_ind]) / self.max_r
        else:
            inputs = random_wrap(self.noise[[np.random.choice(self.noise_inds)]])
            target_c = self._from_numpy(np.asarray([0]))
            target_r = self._from_numpy(np.asarray([-1]))

        # convert to spectrogram
        inputs = self._from_numpy(to_spect(inputs).copy())

        # transform
        if self.transform is not None:
            inputs = self.transform(inputs)

        # squeeze
        if self.squeeze:
            inputs = inputs.squeeze()
    
        return inputs, target_c, target_r, t_ind

    def __len__(self):
        return self.data.shape[0]
    
    def _from_numpy(self, tensor):
        return torch.from_numpy(tensor).float()

    def _load_h5(self):
        file = h5py.File(config['dataset']['data_directory'] + "/VDS_main.h5", 'r', libver='latest')
        file_n = h5py.File(config['dataset']['data_directory'] + "/" + config['dataset']['noise_data'])
        return file['data'], file['labels'], file['range_labels'], file_n['data']

class SimDataRangeClassifyEvaluateLocalize(data.Dataset):
    """
    Dataset class for simulated data classification and ranging tasks. This dataloader is stochastic in that it, with 50-50 chance, chooses to 
    load noise or the signal. If it loads the signal it randomly chooses a TOSSIT to load the signal from. If it loads noise, the signal is
    randomly rotated.

    ...

    Attributes
    ----------
    data : HDF5 file
        pointer to (simulated) data-containing HDF5 file
    labels : HDF5 file
        pointer to (simulated) data labels HDF5 file
    range_labels : array-like
        the range of the source from each TOSSIT
    noise : HDF5 file
        pointer to (experimental) noise signals in HDF5 file
    num_noises : int
        the total number of experimental noise signals in 'noise'
    noise_inds : array-like
        array of indices for 'noise' which correspond to 'split' 
    size : tuple
        spectrogram shape
    num_TOSSITs : int
        number of TOSSITs used
    max_r : float
        maximum range in the dataset for scaling
    transform : PyTorch Compose object
        desired data transformations
    squeeze : bool
        whether or not to eliminate the singleton channel dimension
    track_TOSSIT : bool
        whether or not we use deterministic TOSSIT selection for each data example
    TOSSIT_LUT : array-like
        map from data index to TOSSIT to take data from 
    """

    def __init__(self, split, transform=None, squeeze=False):
        """
        Construct attributes and grab reference to data file

        Parameters
        ----------
        split : string
            which split to draw data from
        transform : PyTorch Compose object
            desired data transformations
        squeeze : bool
            whether or not to eliminate the singleton channel dimension 
        track_TOSSIT : bool
            whether or not we use deterministic TOSSIT selection for each data example
        """

        super(SimDataRangeClassifyEvaluateLocalize, self).__init__()

        # make sure split is valid
        assert split in ['train', 'test', 'val'], "'split' must be one of 'train', 'test', 'val'"

        # get data file
        self.data, self.labels, self.range_labels, self.noise = self._load_h5()

        # get number of noise examples
        self.num_noises = self.noise.shape[0]
        self.noise_inds = np.load(config['dataset']['data_directory'] + f'/{split}_noise_indices.npy', allow_pickle=True)

        # for scaling location labels
        self.max_r = config['scaling']['max_r']

        # get imsize
        self.num_TOSSITs = self.data.shape[1]
        self.size = to_spect(self.data[0]).shape[2:]

        self.transform = transform
        self.squeeze = squeeze

    def __getitem__(self, idx):

        if np.random.choice([0,1]):
            inputs = self.data[idx]
            target_c = self._from_numpy(np.asarray([1]))
            target_r = self._from_numpy(self.range_labels[idx])
            x_target = self._from_numpy(self.labels[[idx],1])
            y_target = self._from_numpy(self.labels[[idx],0])
        else:
            ninds = np.random.choice(self.noise_inds, size=self.num_TOSSITs)
            inputs = random_wrap(self.noise[ninds])
            target_r = self._from_numpy(-1*np.ones(self.num_TOSSITs,))
            target_c = self._from_numpy(np.asarray([0]))
            x_target = self._from_numpy(np.asarray([-1]))
            y_target = self._from_numpy(np.asarray([-1]))

        # convert to spectrogram
        inputs = self._from_numpy(to_spect(inputs).copy())

        # transform
        if self.transform is not None:
            for i in range(self.num_TOSSITs):
                inputs[i,...] = self.transform(inputs[i,...])

        # squeeze
        if self.squeeze:
            inputs = inputs.squeeze()
    
        return inputs, target_c, target_r, x_target, y_target

    def __len__(self):
        return self.data.shape[0]
    
    def _from_numpy(self, tensor):
        return torch.from_numpy(tensor).float()

    def _load_h5(self):
        file = h5py.File(config['dataset']['data_directory'] + "/VDS_main.h5", 'r', libver='latest')
        file_n = h5py.File(config['dataset']['data_directory'] + "/" + config['dataset']['noise_data'])
        return file['data'], file['labels'], file['range_labels'], file_n['data'][:]

###########################################################
#                Dataloader Constructors                  #
###########################################################

def get_dataloaders_range_classify(splits, batch_size, shuffle=True, drop_last=False, transform=None, squeeze=False, num_workers=10, pin_memory=False):
    """
    Construct dictionary of dataloaders for splits ('train', 'val', 'test'), for the range/classification tasks.

    Parameters
    ----------
    splits : list[str]
        list of desires splits to include in dataloaders dict
    batch_size : int
        number of elements per batch
    shuffle : bool
        whether or not to shuffle the training set
    transform : PyTorch Compose object
        desired data transformations
    squeeze : bool
        whether or not to eliminate the singleton channel dimension     

    Returns
    -------
    dataloaders : dict
        dictionary of dataloaders for each split
    """
    
    assert all(x in ['train', 'val', 'test'] for x in splits), "Valid splits are 'train', 'val', 'test'"

    # prepare transforms
    data_transform = dict()
    for split in splits:
        if split == 'train':
            data_transform[split] = transform[split] if transform is not None else transform
        else:
            data_transform[split] = transform['eval'] if transform is not None else transform

    # prepare datasets
    datasets = {x : SimDataRangeClassify(split=x, transform=data_transform[x], squeeze=squeeze) for x in data_transform.keys()}

    # prepare dataloaders
    dataloaders = {x : data.DataLoader(datasets[x], num_workers=num_workers, batch_sampler=H5BatchSampler(split=x, batch_size=batch_size, drop_last=drop_last, shuffle=False if x != 'train' else shuffle), pin_memory=pin_memory) for x in data_transform.keys()}

    # return dataloaders
    return dataloaders

def get_dataloaders_range_classify_eval_loc(splits, batch_size, drop_last=False, transform=None, squeeze=False, num_workers=10, pin_memory=False):
    
    assert all(x in ['val', 'test'] for x in splits), "Valid splits are 'val', 'test'"

    # prepare transforms
    data_transform = dict()
    for split in splits:
        data_transform[split] = transform['eval'] if transform is not None else transform

    # prepare datasets
    datasets = {x : SimDataRangeClassifyEvaluateLocalize(split=x, transform=data_transform[x], squeeze=squeeze) for x in data_transform.keys()}

    # prepare dataloaders
    dataloaders = {x : data.DataLoader(datasets[x], num_workers=num_workers, batch_sampler=H5BatchSampler(split=x, batch_size=batch_size, drop_last=drop_last, shuffle=False), pin_memory=pin_memory) for x in data_transform.keys()}

    # return dataloaders
    return dataloaders