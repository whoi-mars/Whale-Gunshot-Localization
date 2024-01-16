import torch
import torch.nn as nn

class FCN(nn.Module):
    """
        Fully Connected network
    """

    def __init__(self, n_hidden, h_size, i_size, o_size):
        """
            Initialize
        """
        super(FCN, self).__init__()
        assert n_hidden >= 1, "n_hidden must be >= 1"
        
        linear_list = [nn.Linear(i_size, h_size)]
        for _ in range(n_hidden-1):
            linear_list += [nn.Linear(h_size, h_size)]
        linear_list += [nn.Linear(h_size, o_size)]
        self.network = nn.Sequential(*linear_list)

    def forward(self, x):
        return self.network(x)