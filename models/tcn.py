# Code adapted from https://github.com/locuslab/TCN

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

##################################################################
#                       TCN Building Blocks                      #
##################################################################

class Chomp1d(nn.Module):
    """
    Chop a portion of the last channel of the tensor.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN residual block.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    Standard TCN.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class FusionTemporalConvNet(nn.Module):
    """
    TCN with multi-input fusion.
    """

    def __init__(self, num_inputs, input_channels, num_channels, kernel_size=2, dropout=0.2):
        super(FusionTemporalConvNet, self).__init__()
        # layers to fuse inputs
        self.fusion_layers = nn.ModuleList([TemporalBlock(input_channels, num_channels[0], kernel_size, stride=1, dilation=1,
                            padding=(kernel_size-1) * 1, dropout=dropout) for _ in range(num_inputs)])
        self.num_inputs = num_inputs
        # tcn
        layers = []
        num_levels = len(num_channels)
        for i in range(1, num_levels):
            dilation_size = 2 ** i
            in_channels = self.num_inputs * num_channels[0] if i == 1 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # fuse inputs
        fusion_outs = [self.fusion_layers[i](x[:,i,...]) for i in range(self.num_inputs)]
        # fusion_out1 = self.fusion_layers[0](x[:,0,...])
        # fusion_out2 = self.fusion_layers[1](x[:,1,...])
        # fusion_out3 = self.fusion_layers[2](x[:,2,...])
        
        # concatenate channels
        # x = torch.cat((fusion_out1, fusion_out2, fusion_out3), dim=1)
        x = torch.cat(fusion_outs, dim=1)

        # run through rest of TCN layers
        return self.network(x)

##################################################################
#                           TCN Models                           #
##################################################################

class FusionTCN(nn.Module):
    """
    TCN with multi-input fusion and a linear output layer.
    """

    def __init__(self, num_inputs, input_size, output_size, num_channels, kernel_size, dropout):
        super(FusionTCN, self).__init__()
        self.tcn = FusionTemporalConvNet(num_inputs, input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        x = self.tcn(inputs)
        return self.linear(x[:,:,-1])