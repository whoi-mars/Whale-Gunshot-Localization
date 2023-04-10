import yaml

import torch
from torch.utils import data
import h5py
import numpy as np

from utils.transformations import to_spect
from datasets.samplers import H5BatchSampler, UniformGridH5BatchSampler

# load config file
with open("config.yaml", 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.Loader)

class SimData(data.Dataset):
    """
    Dataset class for simulated data.

    ...

    Attributes
    ----------
    inputs : HDF5 file
        pointer to (simulated) data-containing HDF5 file
    size : tuple
        spectrogram shape
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
    Dataset class for simulated data.

    ...

    Attributes
    ----------
    inputs : HDF5 file
        pointer to (simulated) data-containing HDF5 file
    size : tuple
        spectrogram shape
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

def get_dataloaders(splits, batch_size, shuffle=True, drop_last=False, transform=None, squeeze=False, num_workers=10, pin_memory=False):
    """
    Construct dictionary of dataloaders for splits ('train', 'val', 'test').

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

    assert all(x in ['train', 'val', 'test'] for x in splits), "Valid splits are 'train', 'val', and 'test'"

    # prepare transforms
    data_transform = dict()
    for split in splits:
        if split == 'train':
            data_transform[split] = transform['train'] if transform is not None else transform
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
    Construct dictionary of dataloaders for splits ('train', 'val', 'test'). The 'train' split uses the UniformGridH5BatchSampler, and
    the others use the nominal H5BatchSampler. By default, the train split shuffles and the others don't.

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

    assert all(x in ['train', 'val', 'test'] for x in splits), "Valid splits are 'train', 'val', and 'test'"

    # prepare transforms
    data_transform = dict()
    for split in splits:
        if split == 'train':
            data_transform[split] = transform['train'] if transform is not None else transform
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
    Construct the dataloader for experimental data. The data is not shuffled.

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