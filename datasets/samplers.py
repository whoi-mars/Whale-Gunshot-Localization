import math

import yaml
import numpy as np

import torch
from torch.utils import data

# load config file
with open("config.yaml", 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.Loader)

class H5BatchSampler(data.Sampler):
    """
    Batch sampler which samples from a set of indices determined by the specified split.

    ...

    Attributes
    ----------
    idx : array-like
        index list to sample from
    batch_size : int
        samples per batch
    shuffle : bool
        whether or not to shuffle split index list
    """

    def __init__(self, split, batch_size, drop_last=False, shuffle=False):
        """
        Construct attributes.

        Parameters
        ----------
        split : str
            name of the data split ('train', 'test', or 'val')
        batch_size : int
            samples per batch
        drop_last : bool
            whether or not to drop remainder after last batch is taken
        shuffle : bool
            whether or not to shuffle split index list
        """
        self.idx = np.load(config['dataset']['data_directory'] + f'/{split}_indices.npy', allow_pickle=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.__batch_length = len(self.idx) // self.batch_size if self.drop_last else math.ceil(len(self.idx) / self.batch_size)

    def _chunk(self, indices, size):
        """
        Splits indices into groups with number of elements specified by size and a remainder group.
        """

        return torch.split(torch.tensor(indices), size)

    def __iter__(self):
        """
        Make batches iterator.
        """

        if self.shuffle:
            np.random.shuffle(self.idx)

        all_batches = list(self._chunk(self.idx, self.batch_size))
        if self.drop_last:
            all_batches.pop()
        all_batches = [batch.tolist() for batch in all_batches]

        return iter(all_batches)

    def __len__(self):
        return self.__batch_length