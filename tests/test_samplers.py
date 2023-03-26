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

    def test_train_indices(self):
        if os.path.isfile(config['dataset']['data_directory'] + '/train_indices.npy'):
            h5bs = H5BatchSampler(split='train', batch_size=1)
            train_indices = np.load(config['dataset']['data_directory'] + '/train_indices.npy', allow_pickle=True)
            h5bs_list = [idx for batch in list(h5bs) for idx in batch]
            assert(all(x in h5bs_list for x in train_indices))
        else:
            warnings.warn("No train indices saved.")
            assert(True)

    def test_val_indices(self):
        if os.path.isfile(config['dataset']['data_directory'] + '/val_indices.npy'):
            h5bs = H5BatchSampler(split='val', batch_size=1)
            val_indices = np.load(config['dataset']['data_directory'] + '/val_indices.npy', allow_pickle=True)
            h5bs_list = [idx for batch in list(h5bs) for idx in batch]
            assert(all(x in h5bs_list for x in val_indices))
        else:
            warnings.warn("No validation indices saved.")
            assert(True)

    def test_test_indices(self):
        if os.path.isfile(config['dataset']['data_directory'] + '/test_indices.npy'):
            h5bs = H5BatchSampler(split='test', batch_size=1)
            test_indices = np.load(config['dataset']['data_directory'] + '/test_indices.npy', allow_pickle=True)
            h5bs_list = [idx for batch in list(h5bs) for idx in batch]
            assert(all(x in h5bs_list for x in test_indices))
        else:
            warnings.warn("No test indices saved.")
            assert(True)
    
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
        next(iter(h5bs))
        return True

# TODO
class TestUniformGridH5BatchSampler(unittest.TestCase):

    def test_test(self):
        h5bs = UniformGridH5BatchSampler(split='train', batch_size=128, grid_dims=(10, 10))
        print(next(iter(h5bs)))
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