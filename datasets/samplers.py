import math

import yaml
import numpy as np
import h5py
from scipy.stats import binned_statistic_2d
import cvxpy as cp
import matplotlib.pyplot as plt

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
        self.__batch_length = self.num_samples // self.batch_size if self.drop_last else math.ceil(self.num_samples / self.batch_size)
        self._divisible = (self.num_samples % self.batch_size == 0)

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
        if self.drop_last and not self._divisible:
            all_batches.pop()
        all_batches = [batch.tolist() for batch in all_batches]

        return iter(all_batches)

    def __len__(self):
        return self.__batch_length

class ImbalancedH5BatchSampler(data.Sampler):
    """
    Batch sampler which divides the X/Y locations into 2d bins and samples uniformly
    among the bins.

    ...

    Attributes
    ----------
    idx : array-like
        index list to sample from
    batch_size : int
        samples per batch
    drop_last : bool
        whether or not to drop remainder after last batch is taken
    num_sampes : int
        number of samples in split
    bin_lists : List[array-like]
        a list of lists where each of the sublist contain indices of X/Y labels
        which correspond to a particular bin
    bin_samp_len_list : List[int]
        number of indices to sample from each bin
    """

    def __init__(self, split, batch_size, grid_dims, drop_last=False):
        """
        Construct attributes.

        Parameters
        ----------
        split : str
            name of the data split ('train', 'test', or 'val')
        batch_size : int
            samples per batch
        grid_dims : tuple
            2-D tuple that specifies how the grid cells are layed out
        drop_last : bool
            whether or not to drop remainder after last batch is taken
        """

        self.idx = np.load(config['dataset']['data_directory'] + f'/{split}_indices.npy', allow_pickle=True)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(self.idx)
        self.__batch_length = self.num_samples // self.batch_size if self.drop_last else math.ceil(self.num_samples / self.batch_size)
        self._divisible = (self.num_samples % self.batch_size == 0)

        self.bin_lists, self.bin_samp_len_list = self._bin_examples(grid_dims)

        # make sure grid_dims is valid
        assert isinstance(grid_dims, tuple) and len(grid_dims) == 2, "grid_dims must be a tuple of length 2"

        # make sure a batch can have samples from all bins
        assert (self.batch_size / math.prod(grid_dims)) > 1, "Number of grid cells is larger than batch size"

    def _chunk(self, indices, size):
        """
        Splits indices into groups with number of elements specified by size and a remainder group.
        """

        return torch.split(torch.tensor(indices), size)

    def _bin_examples(self, grid_dims):
        """
        Generate lists of which indices in split belong to each bin. Also calculate how many samples should be
        drawn from each list to accomplish approximatly uniform sampling among the bins.

        Parameters
        ----------
        grid_dims : tuple
            number of bins in the X/Y dimensions to overlay onto the locations grid, (# Y, # X)

        Returns
        -------
        bin_lists : List[array-like]
            List of sublists, each of which contain indices of the split which correspond to a particular X/Y location bin
        bin_samp_len_list : List[int]
            List of integers which represent how many samples should be drawn from each bin list to get uniform sampling among
            the bins. Note that any "extra" samples are put into the last element, but in the __iter__ method we randomize which
            bin list is drawn last
        """

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
        last_samp_num = samp_num if (self.num_samples % samp_num == 0) else (samp_num + self.num_samples % samp_num) 
        bin_samp_len_list[-1] = last_samp_num

        return bin_lists, bin_samp_len_list

    def __iter__(self):
        """
        Make batches iterator
        """

        # choose order to sample from bins
        bin_order = np.arange(len(self.bin_lists))
        np.random.shuffle(bin_order)
        
        # sample epoch
        all_batches = np.concatenate([np.random.choice(self.bin_lists[i], size=self.bin_samp_len_list[i]) for i in bin_order])

        # shuffle
        np.random.shuffle(all_batches)
        
        # make batches
        all_batches = list(self._chunk(self.idx, self.batch_size))
        if self.drop_last and not self._divisible:
            all_batches.pop()
        all_batches = [batch.tolist() for batch in all_batches]

        return iter(all_batches)
    
    def __len__(self):
        return self.__batch_length

class UniformGridH5BatchSampler(data.Sampler):
    """
    Batch sampler which divides the X/Y locations into 2d bins and samples to
    make the distribution of X/Y locations as uniform as possible.

    ...

    Attributes
    ----------
    idx : array-like
        index list to sample from
    batch_size : int
        samples per batch
    drop_last : bool
        whether or not to drop remainder after last batch is taken
    num_sampes : int
        number of samples in split
    bin_lists : List[array-like]
        a list of lists where each of the sublist contain indices of X/Y labels
        which correspond to a particular bin
    sample_map : array-like
        array containing how many samples to draw from each list in bin_lists
    """

    def __init__(self, split, batch_size, grid_dims, drop_last=False):
        """
        Construct attributes.

        Parameters
        ----------
        split : str
            name of the data split ('train', 'test', or 'val')
        batch_size : int
            samples per batch
        grid_dims : tuple
            2-D tuple that specifies how the grid cells are layed out
        drop_last : bool
            whether or not to drop remainder after last batch is taken
        """

        self.idx = np.load(config['dataset']['data_directory'] + f'/{split}_indices.npy', allow_pickle=True)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(self.idx)
        self.__batch_length = self.num_samples // self.batch_size if self.drop_last else math.ceil(self.num_samples / self.batch_size)
        self._divisible = (self.num_samples % self.batch_size == 0)

        self.sample_map, self.bin_lists = self._calc_bin_samples(grid_dims)

        # make sure grid_dims is valid
        assert isinstance(grid_dims, tuple) and len(grid_dims) == 2, "grid_dims must be a tuple of length 2"

        # make sure a batch can have samples from all bins
        assert (self.batch_size / math.prod(grid_dims)) > 1, "Number of grid cells is larger than batch size"

    def _chunk(self, indices, size):
        """
        Splits indices into groups with number of elements specified by size and a remainder group.
        """

        return torch.split(torch.tensor(indices), size)

    def _calc_bin_samples(self, grid_dims):
        """
        Solves a simple convex optimization problem to figure out how to
        sample from each of the grid cells in order to make the distributions
        of X and Y locations as uniform as possible. The problem we solve is

        min c
        s.t. Ax == b
             x[zero_inds] == 0
             x[non_zero_inds] >= x_target - c*x_target
             x[non_zero_inds] <= x_target + c*x_target.

        This constrains the sums of every row/column to form approximately uniform distributions
        w.r.t. X/Y locations, ensures that empty bins remain at zero, and try to make the number of
        samples taken from each bin to be as similar as possible.

        Parameters
        ----------
        grid_dims : tuple
            number of bins in the X/Y dimensions to overlay onto the locations grid, (# Y, # X) 
        
        Returns
        -------
        sample_map : array-like
            list how how many samples to draw from each binned index list in 'bin_lists'
        bin_lists : List[array-like]
            list of sublists, each of which contain indices of the split which correspond to a particular X/Y location bin
        """
        
        # get indices to sort idx
        idx_sort = self.idx.argsort()

        # get indices to unsort idx
        idx_unsort = np.empty_like(idx_sort)
        idx_unsort[idx_sort] = np.arange(idx_sort.size)

        # get location labels
        with h5py.File(config['dataset']['data_directory'] + "/VDS_main.h5", 'r') as f:
            y_locs = f['labels'][self.idx[idx_sort],0][idx_unsort]
            x_locs = f['labels'][self.idx[idx_sort],1][idx_unsort]

        # plt.figure()
        # plt.hist(y_locs, bins='auto')
        # plt.savefig("y_orig.jpg")
        # plt.figure()
        # plt.hist(x_locs, bins='auto')
        # plt.savefig("x_orig.jpg")

        # get bins
        ret = binned_statistic_2d(y_locs,
                                  x_locs,
                                  None,
                                  'count',
                                  bins=list(grid_dims[::-1]))
        
        # group indices by bin
        bins = ret.binnumber
        unique_bins = sorted(set(bins))
        bin_lists = [self.idx[bins == b] for b in unique_bins]
        
        # get counts statistic
        counts = ret.statistic

        # flatten counts
        counts_flat = counts.flatten('C')

        # construct main equality constraint matrix
        inds = np.arange(len(counts_flat)).reshape(counts.shape)
        inds = np.concatenate((inds, inds.T), axis=0)

        A = np.zeros((sum(counts.shape), math.prod(counts.shape)))
        b = np.zeros((sum(counts.shape),))

        for i in range(A.shape[0]):
            A[i, inds[i,:]] = counts_flat[inds[i,:]]

        b[:counts.shape[0]-1] = self.num_samples // counts.shape[0]
        b[counts.shape[0]-1] = b[counts.shape[0]-2] + (self.num_samples % counts.shape[0])
        b[counts.shape[0]:counts.shape[0] + counts.shape[1]-1] = self.num_samples // counts.shape[1]
        b[counts.shape[0] + counts.shape[1]-1] = b[counts.shape[0] + counts.shape[1]-2] + (self.num_samples % counts.shape[1])

        # set up optimization problem
        # get zero and nonzero indieces of counts
        zero_inds = np.nonzero(counts_flat == 0)
        non_zero_inds = np.nonzero(counts_flat)

        # decision variables
        x = cp.Variable(shape=len(counts_flat))
        c = cp.Variable(shape=1)

        # set up constraints
        num_samp_target = counts_flat.sum() / ((counts_flat != 0).sum())
        counts_flat[counts_flat == 0] = 1
        x_target = num_samp_target / counts_flat
        constraints = [A@x == b,
                       x[zero_inds] == 0,
                       x[non_zero_inds] <= (x_target[non_zero_inds] + c*x_target[non_zero_inds]),
                       x[non_zero_inds] >= (x_target[non_zero_inds] - c*x_target[non_zero_inds])]  

        # set up objective
        obj = cp.Minimize(c)

        # solve problem
        prob = cp.Problem(obj, constraints)
        res = prob.solve()
        
        # get sample numbers
        sample_map = np.ceil(counts_flat*x.value)

        # eliminate extra
        extra = sample_map.sum() - counts.sum()
        while extra > 0:
            ind_max = sample_map.argmax()
            sample_map[ind_max] -= 1
            extra -= 1

        sample_map = np.delete(sample_map, np.where(counts_flat == 1)).astype(int)

        return sample_map, bin_lists

    def __iter__(self):
        """
        Make batches iterator
        """

        # sample epoch
        all_batches = np.concatenate([np.random.choice(bl, size=samps) for (bl, samps) in zip(self.bin_lists, self.sample_map)])

        # get location labels
        # with h5py.File(config['dataset']['data_directory'] + "/VDS_main.h5", 'r') as f:
        #     y_locs = f['labels'][:,0]
        #     x_locs = f['labels'][:,1]

        # y_locs = y_locs[all_batches]
        # x_locs = x_locs[all_batches]

        # plt.figure()
        # plt.hist(y_locs, bins='auto')
        # plt.savefig("y.jpg")
        # plt.figure()
        # plt.hist(x_locs, bins='auto')
        # plt.savefig("x.jpg")

        # shuffle
        np.random.shuffle(all_batches)
        
        # make batches
        all_batches = list(self._chunk(self.idx, self.batch_size))
        if self.drop_last and not self._divisible:
            all_batches.pop()
        all_batches = [batch.tolist() for batch in all_batches]

        return iter(all_batches)

    def __len__(self):
        return self.__batch_length