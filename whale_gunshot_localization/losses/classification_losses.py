import torch
import torch.nn as nn

class SigBCE(nn.Module):
    """
    Wrapper for BCE loss and sigmoid activation function
    """
    def __init__(self):
        super(SigBCE, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.BCE = nn.BCELoss()
    
    def forward(self, outputs, targets):
        return self.BCE(self.sigmoid(outputs), targets)