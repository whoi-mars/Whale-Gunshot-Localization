import torch.nn as nn

class LocMSELoss(nn.Module):
    """
    Summed MSE loss for the X and Y location components.

    ...

    Attributes
    ----------
    MSE : nn.Module
        pytorch MSE loss functio module
    """

    def __init__(self):
        """
        Construct MSE module
        """

        super(LocMSELoss, self).__init__()
        self.MSE = nn.MSELoss()

    def forward(self, outputs, x_targets, y_targets):
        """
        Calculate loss.

        Parameters
        ----------
        outputs : array-like
            predicted X and Y location for a batch of inputs
        x_targets : array-like
            true X locations for a batch of data
        y_targets : array-like
            true Y locations for a batch of data

        Returns
        -------
        float
            calculated loss
        """

        # TODO: combine both targets into a single MSE vector?
        x_loss = self.MSE(outputs[:,0], x_targets)
        y_loss = self.MSE(outputs[:,1], y_targets)
        return x_loss + y_loss