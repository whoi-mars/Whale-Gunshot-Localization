import math

import yaml
import numpy as np
import h5py
from scipy.stats import binned_statistic_2d

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
    drop_last : bool
        whether or not to drop remainder after last batch is taken
    num_sampes : int
        number of samples in split
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
        self.num_samples = len(self.idx)
        self.__batch_length = len(self.idx) // self.batch_size if self.drop_last else math.ceil(len(self.idx) / self.batch_size)
        self.divisible = (self.num_samples % self.batch_size == 0)

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
        if self.drop_last and not self.divisible:
            all_batches.pop()
        all_batches = [batch.tolist() for batch in all_batches]

        return iter(all_batches)

    def __len__(self):
        return self.__batch_length

class ImbalancedH5BatchSampler(data.Sampler):

    def __init__(self, split, batch_size, grid_dims, drop_last=False):
        
        self.idx = np.load(config['dataset']['data_directory'] + f'/{split}_indices.npy', allow_pickle=True)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(self.idx)
        self.__batch_length = len(self.idx) // self.batch_size if self.drop_last else math.ceil(len(self.idx) / self.batch_size)
        self.divisible = (self.num_samples % self.batch_size == 0)

        self.bin_lists, self.bin_samp_len_list = self._bin_examples(grid_dims)

    def _chunk(self, indices, size):
        """
        Splits indices into groups with number of elements specified by size and a remainder group.
        """

        return torch.split(torch.tensor(indices), size)

    def _bin_examples(self, grid_dims):
        
        # get indices to sort idx
        idx_sort = self.idx.argsort()

        # get indices to unsort idx
        idx_unsort = np.empty_like(idx_sort)
        idx_unsort[idx_sort] = np.arange(idx_sort.size)

        # get location labels
        with h5py.File(config['dataset']['data_directory'] + "/VDS_main.h5", 'r') as f:
            y_locs = f['labels'][self.idx[idx_sort],0][idx_unsort]
            x_locs = f['labels'][self.idx[idx_sort],1][idx_unsort]

        # get bins
        ret = binned_statistic_2d(y_locs,
                                  x_locs,
                                  None,
                                  'count',
                                  bins=list(grid_dims[::-1]))
        bins = ret.binnumber

        # group indices by bin
        unique_bins = sorted(set(bins))
        bin_lists = [self.idx[bins == b] for b in unique_bins]

        # get number of samples from each bin to make an epoch from
        samp_num = self.num_samples // len(unique_bins)
        bin_samp_len_list = np.ones((len(unique_bins,),), dtype=int) * samp_num
        last_samp_num = samp_num if self.num_samples % samp_num == 0 else samp_num + self.num_samples % samp_num 
        bin_samp_len_list[-1] = last_samp_num

        return bin_lists, bin_samp_len_list

    def __iter__(self):
        
        # choose order to sample from bins
        bin_order = np.arange(len(self.bin_lists))
        np.random.shuffle(bin_order)
        
        # sample epoch
        all_batches = np.concatenate([np.random.choice(self.bin_lists[i], size=self.bin_samp_len_list[i]) for i in bin_order])

        # shuffle
        np.random.shuffle(all_batches)
        
        # make batches
        all_batches = list(self._chunk(self.idx, self.batch_size))
        if self.drop_last and not self.divisible:
            all_batches.pop()
        all_batches = [batch.tolist() for batch in all_batches]

        return iter(all_batches)
    
    def __len__(self):
        return self.__batch_length