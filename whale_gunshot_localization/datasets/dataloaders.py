import yaml

import torch
from torch.utils import data
import h5py
import numpy as np

from whale_gunshot_localization.utils.transformations import to_spect, random_wrap
from whale_gunshot_localization.datasets.samplers import H5BatchSampler, UniformGridH5BatchSampler
from whale_gunshot_localization import config

###########################################################
#                      Localization                       #
###########################################################

class SimData(data.Dataset):
    """
    Dataset class for simulated data. For the task of predicting the X and Y component of
    source location.

    ...

    Attributes
    ----------
    data : HDF5 file
        pointer to (simulated) data-containing HDF5 file
    labels : HDF5 file
        pointer to (simulated) data labels HDF5 file
    size : tuple
        spectrogram shape
    num_TOSSITs : int
        number of TOSSITs used
    max_x : float
        largest x location
    max_y : float
        largest y location
    transform : PyTorch Compose object
        desired data transformations
    squeeze : bool
        whether or not to eliminate the singleton channel dimension 
    """

    def __init__(self, transform=None, squeeze=False):
        """
        Construct attributes and grab reference to data file

        Parameters
        ----------
        transform : PyTorch Compose object
            desired data transformations
        squeeze : bool
            whether or not to eliminate the singleton channel dimension 
        """

        super(SimData, self).__init__()

        # get data file
        self.data, self.labels = self._load_h5()

        # get imsize
        self.num_TOSSITs = self.data.shape[1]
        self.size = to_spect(self.data[0]).shape[2:]

        # for scaling location labels
        self.max_x = config['scaling']['max_x']
        # self.min_x = config['scaling']['min_x']
        self.max_y = config['scaling']['max_y']
        # self.min_y = config['scaling']['min_y']

        self.transform = transform
        self.squeeze = squeeze

    def __getitem__(self, idx):

        # get data
        inputs = self.data[idx]

        # convert to spectrogram
        inputs = self._from_numpy(to_spect(inputs).copy())

        # transform
        if self.transform is not None:
            inputs = self.transform(inputs)

        # squeeze
        if self.squeeze:
            inputs = inputs.squeeze()

        # get labels
        x_target = self._from_numpy(self.labels[[idx],1]) / self.max_x
        y_target = self._from_numpy(self.labels[[idx],0]) / self.max_y
        # x_target = (self._from_numpy(self.labels[[idx],1]) - self.min_x) / (self.max_x - self.min_x)
        # y_target = (self._from_numpy(self.labels[[idx],0]) - self.min_y) / (self.max_y - self.min_y)
        
        return inputs, x_target, y_target

    def __len__(self):
        return self.data.shape[0]
    
    def _from_numpy(self, tensor):
        return torch.from_numpy(tensor).float()

    def _load_h5(self):
        file = h5py.File(config['dataset']['data_directory'] + "/VDS_main.h5", 'r', libver='latest')
        return file['data'], file['labels']

class ExpData(SimData):
    """
    Dataset class for experimental data. For the task of predicting the X and Y component of
    source location.

    ...

    Attributes
    ----------
    data : HDF5 file
        pointer to (simulated) data-containing HDF5 file
    labels : HDF5 file
        pointer to (simulated) data labels HDF5 file
    size : tuple
        spectrogram shape
    num_TOSSITs : int
        number of TOSSITs used
    max_x : float
        largest x location
    max_y : float
        largest y location
    transform : PyTorch Compose object
        desired data transformations
    squeeze : bool
        whether or not to eliminate the singleton channel dimension 
    """

    def _load_h5(self):
        file = h5py.File(config['dataset']['exp_data_directory'] + "/" + config['dataset']['exp_data_file'], 'r', libver='latest')

        # if there is only one sample just load it into RAM and ensure data has a singleton dimension
        if len(file['data'].shape) == 2:
            return np.expand_dims(file['data'][:], axis=0), file['labels'][:]
        else:
            return file['data'], file['labels']

###########################################################
#                      Classification                     #
###########################################################

class SimDataClassify(data.Dataset):
    """
    Dataset class for simulated data classification task. This dataloader is stochastic in that it, with 50-50 chance, chooses to 
    load noise or the signal. If it loads the signal it randomly chooses a TOSSIT to load the signal from. If it load noise, it
    will randomly rotate the noise.

    ...

    Attributes
    ----------
    data : HDF5 file
        pointer to (simulated) data-containing HDF5 file
    labels : HDF5 file
        pointer to (simulated) data labels HDF5 file
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
    transform : PyTorch Compose object
        desired data transformations
    squeeze : bool
        whether or not to eliminate the singleton channel dimension 
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
        """

        super(SimDataClassify, self).__init__()

        # make sure split is valid
        assert split in ['train', 'test', 'val'], "'split' must be one of 'train', 'test', 'val'"

        # get data file
        self.data, self.noise, self.labels = self._load_h5()

        # get number of noise examples
        self.num_noises = self.noise.shape[0]
        self.noise_inds = np.load(config['dataset']['data_directory'] + f'/{split}_noise_indices.npy', allow_pickle=True)

        # get imsize
        self.num_TOSSITs = self.data.shape[1]
        self.size = to_spect(self.data[0]).shape[2:]

        self.transform = transform
        self.squeeze = squeeze

    def __getitem__(self, idx):

        # get data or noise randomly
        if np.random.choice([0, 1]):
            inputs = self.data[idx,[np.random.randint(low=0, high=self.num_TOSSITs)]]
            target = self._from_numpy(np.asarray([1]))
        else:
            inputs = random_wrap(self.noise[[np.random.choice(self.noise_inds)]])
            target = self._from_numpy(np.asarray([0]))

        # convert to spectrogram
        inputs = self._from_numpy(to_spect(inputs).copy())

        # transform
        if self.transform is not None:
            inputs = self.transform(inputs)

        # squeeze
        if self.squeeze:
            inputs = inputs.squeeze()
        
        return inputs, target

    def __len__(self):
        return self.data.shape[0]
    
    def _from_numpy(self, tensor):
        return torch.from_numpy(tensor).float()

    def _load_h5(self):
        file = h5py.File(config['dataset']['data_directory'] + "/VDS_main.h5", 'r', libver='latest')
        file_n = h5py.File(config['dataset']['data_directory'] + "/" + config['dataset']['noise_data'])
        return file['data'], file_n['data'], file['class_labels']

class ExpDataClassify(data.Dataset):
    """
    Dataset class for the detection task for experimental data.

    ...

    Attributes
    ----------
    data : HDF5 file
        pointer to (simulated) data-containing HDF5 file
    labels : HDF5 file
        pointer to (simulated) data labels HDF5 file
    size : tuple
        spectrogram shape
    num_TOSSITs : int
        number of TOSSITs used
    TOSSIT : int
            TOSSIT index to load and classify (indexed from 1)
    transform : PyTorch Compose object
        desired data transformations
    squeeze : bool
        whether or not to eliminate the singleton channel dimension 
    """

    def __init__(self, transform=None, squeeze=False, TOSSIT=1):
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
        TOSSIT : int
            TOSSIT index to load and classify (indexed from 1)
        """

        super(ExpDataClassify, self).__init__()

        # get data file
        self.data, self.labels = self._load_h5()

        # get imsize
        self.num_TOSSITs = self.data.shape[1]
        self.TOSSIT = TOSSIT
        self.size = to_spect(self.data[0]).shape[2:]

        assert self.TOSSIT <= self.num_TOSSITs, "TOSSIT index is larger than the number of TOSSITs available"

        self.transform = transform
        self.squeeze = squeeze

    def __getitem__(self, idx):

        # get data
        inputs = self.data[idx,[self.TOSSIT-1]]
        target = self._from_numpy(np.asarray([1]))

        # convert to spectrogram
        inputs = self._from_numpy(to_spect(inputs).copy())

        # transform
        if self.transform is not None:
            inputs = self.transform(inputs)

        # squeeze
        if self.squeeze:
            inputs = inputs.squeeze()
        
        return inputs, target

    def __len__(self):
        return self.data.shape[0]
    
    def _from_numpy(self, tensor):
        return torch.from_numpy(tensor).float()

    def _load_h5(self):
        file = h5py.File(config['dataset']['exp_data_directory'] + "/" + config['dataset']['exp_data_file'], 'r', libver='latest')

        # if there is only one sample just load it into RAM and ensure data has a singleton dimension
        if len(file['data'].shape) == 2:
            return np.expand_dims(file['data'][:], axis=0), file['labels'][:]
        else:
            return file['data'], file['labels']

###########################################################
#            Localization + Classification                #
###########################################################

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

    def __init__(self, split, transform=None, squeeze=False, track_TOSSIT=False):
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
        self.num_TOSSITs = self.data.shape[1]
        self.size = to_spect(self.data[0]).shape[2:]

        self.track_TOSSIT = track_TOSSIT
        if self.track_TOSSIT:
            self.TOSSIT_LUT = np.random.choice(np.arange(self.num_TOSSITs), size=(self.__len__(),))
        else:
            self.TOSSIT_LUT = None

        self.transform = transform
        self.squeeze = squeeze

    def __getitem__(self, idx):

        # get data or noise randomly
        if np.random.choice([0, 1]):
            t_ind = self.TOSSIT_LUT[idx] if self.track_TOSSIT else np.random.randint(low=0, high=self.num_TOSSITs)
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
    
        return inputs, target_c, target_r

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

class ExpDataRangeClassify(data.Dataset):
    """
    Experimental dataset for the range/classification task.

    ...

    Attributes
    ----------
    data : HDF5 file
        pointer to (simulated) data-containing HDF5 file
    labels : HDF5 file
        pointer to (simulated) data labels HDF5 file
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
    """
    def __init__(self, TOSSIT, transform=None, squeeze=False):
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
        TOSSIT : int
            TOSSIT index to load and classify (indexed from 1)
        """

        super(ExpDataRangeClassify, self).__init__()

        # get data file
        self.data, self.labels = self._load_h5()

        # for scaling location labels
        self.max_r = config['scaling']['max_r']

        # get imsize
        self.num_TOSSITs = self.data.shape[1]
        self.size = to_spect(self.data[0]).shape[2:]
        self.TOSSIT = TOSSIT

        self.transform = transform
        self.squeeze = squeeze

    def __getitem__(self, idx):

        # get data
        inputs = self.data[idx,[self.TOSSIT-1]]
        target_c = self._from_numpy(np.asarray([1]))
        target_r = self._from_numpy(np.asarray(np.nan))

        # convert to spectrogram
        inputs = self._from_numpy(to_spect(inputs).copy())

        # transform
        if self.transform is not None:
            inputs = self.transform(inputs)

        # squeeze
        if self.squeeze:
            inputs = inputs.squeeze()
    
        return inputs, target_c, target_r

    def __len__(self):
        return self.data.shape[0]
    
    def _from_numpy(self, tensor):
        return torch.from_numpy(tensor).float()

    def _load_h5(self):
        file = h5py.File(config['dataset']['exp_data_directory'] + "/" + config['dataset']['exp_data_file'], 'r', libver='latest')

        # if there is only one sample just load it into RAM and ensure data has a singleton dimension
        if len(file['data'].shape) == 2:
            return np.expand_dims(file['data'][:], axis=0), file['labels'][:]
        else:
            return file['data'], file['labels']

###########################################################
#                Dataloader Constructors                  #
###########################################################

def get_dataloaders(splits, batch_size, shuffle=True, drop_last=False, transform=None, squeeze=False, num_workers=10, pin_memory=False):
    """
    Construct dictionary of dataloaders for splits ('train', 'val', 'test', 'stats'), for the localization tasks.

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

    assert all(x in ['train', 'val', 'test', 'stats'] for x in splits), "Valid splits are 'train', 'val', 'test', and stats"

    if 'stats' in splits:
        assert len(splits) == 1, "'stats' must be used alone"

    # prepare transforms
    data_transform = dict()
    for split in splits:
        if split == 'train':
            data_transform[split] = transform[split] if transform is not None else transform
        elif split == 'stats':
            data_transform['train'] = transform[split] if transform is not None else transform
        else:
            data_transform[split] = transform['eval'] if transform is not None else transform

    # prepare datasets
    datasets = {x : SimData(transform=data_transform[x], squeeze=squeeze) for x in data_transform.keys()}

    # prepare dataloaders
    dataloaders = {x : data.DataLoader(datasets[x], num_workers=num_workers, batch_sampler=H5BatchSampler(split=x, batch_size=batch_size, drop_last=drop_last, shuffle=False if x != 'train' else shuffle), pin_memory=pin_memory) for x in data_transform.keys()}

    # return dataloaders
    return dataloaders

def get_dataloaders_uniform_grid_train(splits, batch_size, grid_dims, transform=None, squeeze=False, num_workers=10):
    """
    Construct dictionary of dataloaders for splits ('train', 'val', 'test', 'stats'). The 'train' split uses the UniformGridH5BatchSampler, and
    the others use the nominal H5BatchSampler. By default, the train split shuffles and the others don't. These dataloaders are for the localization tasks.

    Parameters
    ----------
    splits : list[str]
        list of desires splits to include in dataloaders dict
    batch_size : int
        number of elements per batch
    transform : PyTorch Compose object
        desired data transformations
    squeeze : bool
        whether or not to eliminate the singleton channel dimension
    num_workers : int
        number of workers for the dataloader        

    Returns
    -------
    dataloaders : dict
        dictionary of dataloaders for each split
    """

    assert all(x in ['train', 'val', 'test'] for x in splits), "Valid splits are 'train', 'val', 'test', and 'stats'"

    if 'stats' in splits:
        assert len(splits) == 1, "'stats' must be used alone"

    # prepare transforms
    data_transform = dict()
    for split in splits:
        if split == 'train':
            data_transform[split] = transform[split] if transform is not None else transform
        elif split == 'stats':
            data_transform['train'] = transform[split] if transform is not None else transform
        else:
            data_transform[split] = transform['eval'] if transform is not None else transform

    # prepare datasets
    datasets = {x : SimData(transform=data_transform[x], squeeze=squeeze) for x in data_transform.keys()}

    # prepare dataloaders
    dataloaders = dict()
    for x in data_transform.keys():
        if x == 'train':
            dataloaders[x] = data.DataLoader(datasets[x], num_workers=num_workers, batch_sampler=UniformGridH5BatchSampler(split=x, batch_size=batch_size, grid_dims=grid_dims))
        else:
            dataloaders[x] = data.DataLoader(datasets[x], num_workers=num_workers, batch_sampler=H5BatchSampler(split=x, batch_size=batch_size, shuffle=False))

    # return dataloaders
    return dataloaders

def get_exp_data_dataloader(batch_size, transform=None, squeeze=False, num_workers=10):
    """
    Construct the localization dataloaders for experimental data. The data is not shuffled.

    Parameters
    ----------
    batch_size : int
        number of elements per batch
    transform : PyTorch Compose object
        desired data transformations
    squeeze : bool
        whether or not to eliminate the singleton channel dimension
    num_workers : int
        number of workers for the dataloader

    Returns
    -------
    dataloaders : torch.utils.data.DataLoader
        dictionary of dataloaders for each split 
    """

    # prepare transform
    data_transform = transform['eval'] if transform is not None else transform

    # prepare dataset
    dataset = ExpData(transform=data_transform, squeeze=squeeze)

    # return dataloader
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def get_dataloaders_classify(splits, batch_size, shuffle=True, drop_last=False, transform=None, squeeze=False, num_workers=10, pin_memory=False):
    """
    Construct dictionary of dataloaders for splits ('train', 'val', 'test'), for the classification task.

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
    datasets = {x : SimDataClassify(split=x, transform=data_transform[x], squeeze=squeeze) for x in data_transform.keys()}

    # prepare dataloaders
    dataloaders = {x : data.DataLoader(datasets[x], num_workers=num_workers, batch_sampler=H5BatchSampler(split=x, batch_size=batch_size, drop_last=drop_last, shuffle=False if x != 'train' else shuffle), pin_memory=pin_memory) for x in data_transform.keys()}

    # return dataloaders
    return dataloaders

def get_exp_data_dataloader_classify(batch_size, transform=None, squeeze=False, num_workers=10, TOSSIT=1):
    """
    Construct the localization dataloaders for experimental data. The data is not shuffled.

    Parameters
    ----------
    batch_size : int
        number of elements per batch
    transform : PyTorch Compose object
        desired data transformations
    squeeze : bool
        whether or not to eliminate the singleton channel dimension
    num_workers : int
        number of workers for the dataloader

    Returns
    -------
    dataloaders : torch.utils.data.DataLoader
        dictionary of dataloaders for each split 
    """

    # prepare transform
    data_transform = transform['eval'] if transform is not None else transform

    # prepare dataset
    dataset = ExpDataClassify(transform=data_transform, squeeze=squeeze, TOSSIT=TOSSIT)

    # return dataloader
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def get_dataloaders_range_classify(splits, batch_size, track_TOSSIT=False, shuffle=True, drop_last=False, transform=None, squeeze=False, num_workers=10, pin_memory=False):
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
    track_TOSSIT : bool
        whether or not we use deterministic TOSSIT selection for each data example        

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
    datasets = {x : SimDataRangeClassify(split=x, transform=data_transform[x], squeeze=squeeze, track_TOSSIT=track_TOSSIT) for x in data_transform.keys()}

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


def get_exp_data_dataloader_range_classify(batch_size, transform=None, squeeze=False, num_workers=10, TOSSIT=1):
    """
    Construct the range/classification dataloader for experimental data. The data is not shuffled.

    Parameters
    ----------
    batch_size : int
        number of elements per batch
    transform : PyTorch Compose object
        desired data transformations
    squeeze : bool
        whether or not to eliminate the singleton channel dimension
    num_workers : int
        number of workers for the dataloader
    TOSSIT : int
        TOSSIT index to load and classify (indexed from 1)

    Returns
    -------
    dataloaders : torch.utils.data.DataLoader
        dictionary of dataloaders for each split 
    """

    # prepare transform
    data_transform = transform['eval'] if transform is not None else transform

    # prepare dataset
    dataset = ExpDataRangeClassify(transform=data_transform, squeeze=squeeze, TOSSIT=TOSSIT)

    # return dataloader
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)