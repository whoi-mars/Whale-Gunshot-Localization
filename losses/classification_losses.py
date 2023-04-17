import torch
import torch.nn as nn

class SigBCE:
    """
    Wrapper for BCE loss and sigmoid activation function
    """
    def __init__(self):
        self.sigmoid = nn.Sigmoid()
        self.BCE = nn.BCELoss()
    
    def forward(self, outputs, targets):
        return self.BCE(self.sigmoid(outputs), targets)