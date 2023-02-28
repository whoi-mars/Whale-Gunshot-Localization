import yaml

import torch
from torch.utils import data
import h5py

from utils.transformations import to_spect
from datasets.samplers import H5BatchSampler

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
        Construct attributes and grad reference to data file

        Parameters
        ----------
        transform : PyTorch Compose object
            desired data transformations
        squeeze : bool
            whether or not to eliminate the singleton channel dimension 
        """

        super(SimData, self).__init__()

        # get data file
        self.inputs = self._load_h5()

        # get imsize
        self.size = to_spect(self.inputs['data'][[0]]).shape[2:]

        # for scaling location labels
        self.max_x = config['scaling']['max_x']
        self.max_y = config['scaling']['max_y']

        self.transform = transform
        self.squeeze = squeeze

    def __getitem__(self, idx):

        # get data
        inputs = self.inputs['data'][idx]

        # convert to spectrogram
        inputs = self._from_numpy(to_spect(inputs).copy())

        # transform
        if self.transform is not None:
            inputs = self.transform(inputs)

        # squeeze
        if self.squeeze:
            inputs = inputs.squeeze()

        # get labels
        x_target = self._from_numpy(self.inputs['labels'][idx,1])
        y_target = self._from_numpy(self.inputs['labels'][idx,0])

        return inputs, x_target, y_target

    def __len__(self):
        return self.inputs['data'].shape[0]
    
    def _from_numpy(self, tensor):
        return torch.as_tensor(tensor).float()

    def _load_h5(self):
        file = h5py.File(config['dataset']['data_directory'] + "/VDS_main.h5", 'r')
        return dict(data=file['data'], labels=file['labels'])


def get_dataloaders(splits, batch_size, shuffle=True, transform=None, squeeze=False):
    """
    Construct dictionary of dataloaders for every split.

    Parameters
    ----------

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
    data_transform = {x : transform[x] if transform is not None else transform for x in splits}
    
    # prepare datasets
    datasets = {x : SimData(transform=data_transform[x], squeeze=squeeze) for x in data_transform.keys()}

    # prepare dataloaders
    dataloaders = {x : data.DataLoader(datasets[x], num_workers=1, batch_sampler=H5BatchSampler(split=x, batch_size=batch_size, shuffle=False if x != 'train' else shuffle)) for x in data_transform.keys()}

    return dataloaders