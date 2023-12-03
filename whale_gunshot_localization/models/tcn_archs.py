# Code adapted from https://github.com/locuslab/TCN

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from whale_gunshot_localization.models.fcn import FCN

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

class BranchedTemporalConvNet(nn.Module):
    """
    TCN with multi-input fusion.
    """

    def __init__(self, input_channels, num_channels, kernel_size=2, dropout=0.2):
        super(BranchedTemporalConvNet, self).__init__()
        
        # tcn for levels 0 --> num_levels-1
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels-1):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        # network core
        self.network = nn.Sequential(*layers)

        # separate branches for the last layer before output
        i = num_levels-1
        dilation_size = 2 ** i
        in_channels = num_channels[i-1]
        out_channels = num_channels[i]
        self.ends = nn.ModuleList([TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout) for _ in range(2)])

    def forward(self, x):
        # run through rest of TCN layers
        x = self.network(x)

        # return list of results from all output brances
        return [self.ends[i](x) for i in range(2)]

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

##################################################################
#                           TCN Model                            #
##################################################################

class TCNRangeAndClassifyCond(nn.Module):
    """
    TCN with multiple TCN/linear layers.
    """
    
    def __init__(self, input_size, num_channels, kernel_size, dropout, cond_size):
        super(TCNRangeAndClassifyCond, self).__init__()
        self.tcn = BranchedTemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.FCN1 = FCN(n_hidden=1, h_size=512, i_size=num_channels[-1], o_size=1)
        self.FCN2 = FCN(n_hidden=1, h_size=512, i_size=num_channels[-1] + cond_size, o_size=2)

    def forward(self, inputs, cond):
        x = self.tcn(inputs)
        x1 = self.FCN1(x[0][:,:,-1])
        x2 = self.FCN2(torch.cat((x[1][:,:,-1], cond), dim=1))
        return torch.cat((x1, x2), dim=1)

class TCNClassifier(nn.Module):
    """
    TCN with linear output layers.
    """

    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCNClassifier, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, inputs):
        x = self.tcn(inputs)
        return self.linear(x[:,:,-1])

class TCNRangeAndClassify(nn.Module):
    """
    TCN with multiple TCN/linear layers.
    """
    
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCNRangeAndClassify, self).__init__()
        self.tcn = BranchedTemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(num_channels[-1], 1)
        self.linear2 = nn.Linear(num_channels[-1], 2)

    def forward(self, inputs):
        x = self.tcn(inputs)
        x1 = self.linear1(x[0][:,:,-1])
        x2 = self.linear2(x[1][:,:,-1])
        return torch.cat((x1, x2), dim=1)

class TCNRangeAndClassifyUncertain(nn.Module):
    """
    TCN with multiple TCN/linear layers.
    """
    
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCNRangeAndClassifyUncertain, self).__init__()
        self.tcn = BranchedTemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(num_channels[-1], 2)
        self.linear2 = nn.Linear(num_channels[-1], 3)

    def forward(self, inputs):
        x = self.tcn(inputs)
        x1 = self.linear1(x[0][:,:,-1])
        x2 = self.linear2(x[1][:,:,-1])
        return torch.cat((x1, x2), dim=1)