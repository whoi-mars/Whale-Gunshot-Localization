import unittest

import torch

from utils.transformations import Normalize1DChannel

class TestNormalize1DChannel(unittest.TestCase):

    def test_random(self):
        # make random input
        x = torch.randn(3, 1, 20, 60)
        # get means and stds
        mean_list = torch.mean(x, dim=3)
        std_list = torch.std(x, dim=3)
        # normalize
        norm = Normalize1DChannel(mean_list, std_list)
        x = norm(x)
        assert(torch.sum(torch.mean(x, dim=3) > 1e-6).item() == 0)
        assert(torch.sum(torch.std(x, dim=3)).item() == torch.numel(x[:,:,:,0]))