import torch
import torch.nn as nn

class LocMSELoss(nn.Module):
    """
    Summed MSE loss for the X and Y location components.

    ...

    Attributes
    ----------
    MSE : nn.Module
        pytorch MSE loss function module
    """

    def __init__(self, **kwargs):
        """
        Construct MSE module
        """

        super(LocMSELoss, self).__init__()
        self.MSE = nn.MSELoss(**kwargs)

    def forward(self, outputs, x_targets, y_targets):
        """
        Calculate loss.

        Parameters
        ----------
        outputs : array-like
            predicted X and Y location for a batch of inputs (shape: N X 2)
        x_targets : array-like
            true X locations for a batch of data (shape: N X 1)
        y_targets : array-like
            true Y locations for a batch of data (shape: N X 1)

        Returns
        -------
        float
            calculated loss
        """
        
        return self.MSE(outputs, torch.cat((x_targets, y_targets), dim=1))