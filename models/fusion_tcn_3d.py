# Code adapted from https://github.com/locuslab/TCN
import math

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

class ChompBlock1d(nn.Module):
    """
    Chop a portion of the last channel of the tensor. Takes in 
    a batch of 3D tensors.
    """

    def __init__(self, chomp_size):
        super(ChompBlock1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


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

class FusionTemporalBlock(nn.Module):
    """
    TCN residual block to fuse inputs
    """

    def __init__(self, in_width, kernel_size, stride, dilation, padding, dropout):
        super(FusionTemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(in_width, math.ceil(in_width / 2), (1, kernel_size), 
                                            stride=stride, padding=(0, padding), dilation=dilation))
        self.chomp1 = ChompBlock1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(math.ceil(in_width / 2), 1, (1, kernel_size), 
                                            stride=stride, padding=(0, padding), dilation=dilation))
        self.chomp2 = ChompBlock1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(in_width, 1, 1)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # squeeze to get rid of Conv2d channel dimension
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res).squeeze()

class FusionTemporalConvNet(nn.Module):
    """
    TCN with multi-input fusion.
    """

    def __init__(self, num_inputs, num_outputs, input_channels, num_channels, kernel_size=2, dropout=0.2):
        super(FusionTemporalConvNet, self).__init__()

        # save number of input/outputs branches
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        # layers to fuse inputs. each are a separate TemporalBlock.
        self.fusion_layers = nn.ModuleList([TemporalBlock(input_channels, num_channels[0], kernel_size, stride=1, dilation=1,
                            padding=(kernel_size-1) * 1, dropout=dropout) for _ in range(num_inputs)])
        
        # tcn for levels 1 --> num_levels-1
        layers = [FusionTemporalBlock(in_width=self.num_inputs, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1) * 1, dropout=dropout)]
        num_levels = len(num_channels)
        for i in range(1, num_levels-1):
            dilation_size = 2 ** i
            in_channels = num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        # network core
        self.network = nn.Sequential(*layers)

        # separate branche for the last layer before output
        i = num_levels-1
        dilation_size = 2 ** i
        in_channels = num_channels[i-1]
        out_channels = num_channels[i]
        self.ends = nn.ModuleList([TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout) for _ in range(num_outputs)])

    def forward(self, x):
        # fuse inputs
        fusion_outs = [self.fusion_layers[i](x[:,i,...]) for i in range(self.num_inputs)]

        # concatenate channels
        x = torch.stack(fusion_outs, dim=1)

        # run through rest of TCN layers
        x = self.network(x)

        # return list of results from all output brances
        return [self.ends[i](x) for i in range(self.num_outputs)]

##################################################################
#                           TCN Model                            #
##################################################################

class FusionTCN(nn.Module):
    """
    TCN with multi-input fusion and a linear output layer.
    """

    def __init__(self, num_inputs, num_outputs, input_size, output_size, num_channels, kernel_size, dropout):
        super(FusionTCN, self).__init__()
        
        # make sure we can construct the output branches
        assert not output_size % num_outputs, "'output_size' must be divisible by 'num_outputs'."

        self.tcn = FusionTemporalConvNet(num_inputs, num_outputs, input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(num_channels[-1], output_size // num_outputs)
        self.linear2 = nn.Linear(num_channels[-1], output_size // num_outputs)

    def forward(self, inputs):
        x = self.tcn(inputs)
        x1 = self.linear1(x[0][:,:,-1])
        x2 = self.linear2(x[1][:,:,-1])
        return torch.cat((x1, x2), dim=1)