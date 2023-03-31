import os
import warnings

import unittest
import yaml
import numpy as  np

from datasets.samplers import H5BatchSampler, ImbalancedH5BatchSampler, UniformGridH5BatchSampler

# load config file
with open("config.yaml", 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.Loader)

class TestH5BatchSampler(unittest.TestCase):

    def setUp(self):
        
        # load saved indices for splits
        self.inds = [np.load(config['dataset']['data_directory'] + f'/{split}_indices.npy', allow_pickle=True) for split in ['train', 'val', 'test']]
        
        # trim to 1000 so that the tests run quicker
        self.inds = [split[:1000] if len(split) > 1000 else split for split in self.inds]

    def test_train_indices(self):
        h5bs = H5BatchSampler(split='train', batch_size=1)

        # trim object indices to 1000 to match those in setUp
        if len(h5bs.idx) > 1000:
            h5bs.idx = h5bs.idx[:1000]

        train_indices = self.inds[0]
        h5bs_list = [idx for batch in list(h5bs) for idx in batch]
        assert(all(x in h5bs_list for x in train_indices))

    def test_val_indices(self):
        h5bs = H5BatchSampler(split='val', batch_size=1)

        # trim object indices to 1000 to match those in setUp
        if len(h5bs.idx) > 1000:
            h5bs.idx = h5bs.idx[:1000]

        val_indices = self.inds[1]
        h5bs_list = [idx for batch in list(h5bs) for idx in batch]
        assert(all(x in h5bs_list for x in val_indices))

    def test_test_indices(self):
        h5bs = H5BatchSampler(split='test', batch_size=1)

        # trim object indices to 1000 to match those in setUp
        if len(h5bs.idx) > 1000:
            h5bs.idx = h5bs.idx[:1000]

        test_indices = self.inds[2]
        h5bs_list = [idx for batch in list(h5bs) for idx in batch]
        assert(all(x in h5bs_list for x in test_indices))
    
    def test_batch_size(self):
        batch_sizes = [1, 10, 13, 16, 500]
        for bs in batch_sizes:
            with self.subTest(i=bs):
                h5bs_list = list(H5BatchSampler(split='train', batch_size=bs))
                if len(h5bs_list[-1]) < bs:
                    h5bs_list = h5bs_list[:-1]
                assert(all(len(batch) == bs for batch in h5bs_list))

    def test_drop_last(self):
        batch_sizes = [1, 10, 13, 16, 500]
        for bs in batch_sizes:
            with self.subTest(i=bs):
                h5bs = H5BatchSampler(split='train', batch_size=bs, drop_last=True)
                idx = h5bs.idx
                h5bs_list = list(h5bs)
                if bs > len(idx):
                    assert(len(h5bs_list) == 0)
                else:
                    assert(len(h5bs_list[-1]) == bs)

    def test_shuffle(self):
        train_indices = list(np.load(config['dataset']['data_directory'] + '/train_indices.npy', allow_pickle=True))
        h5bs = list(H5BatchSampler(split='train', batch_size=len(train_indices), shuffle=True))[0]
        assert(h5bs != train_indices)

# TODO
class TestImbalancedH5BatchSampler(unittest.TestCase):

    def test_test(self):
        h5bs = ImbalancedH5BatchSampler(split='train', batch_size=128, grid_dims=(10, 10))
        return True

# TODO
class TestUniformGridH5BatchSampler(unittest.TestCase):

    def test_test(self):
        h5bs = UniformGridH5BatchSampler(split='train', batch_size=128, grid_dims=(11, 11))
        return True

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestH5BatchSampler)
    suite.addTest(TestImbalancedH5BatchSampler)
    suite.addTest(TestUniformGridH5BatchSampler)
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())