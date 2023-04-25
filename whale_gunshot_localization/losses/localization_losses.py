import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class LocMSELossVec(nn.Module):
    """
    Summed MSE loss for the X and Y location components. Computed with 
    X/Y locations as a vector.

    ...

    Attributes
    ----------
    MSE : nn.Module
        pytorch MSE loss function module
    """

    def __init__(self):
        """
        Construct MSE module
        """

        super(LocMSELossVec, self).__init__()
        self.MSE = nn.MSELoss()

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

class LocMSELoss(nn.Module):
    """
    Summed MSE loss for the X and Y location components.

    ...

    Attributes
    ----------
    MSE : nn.Module
        pytorch MSE loss function module
    """

    def __init__(self):
        """
        Construct MSE module
        """

        super(LocMSELoss, self).__init__()
        self.MSE = nn.MSELoss(reduction='none')

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
        
        return (self.MSE(outputs[:,[0]], x_targets) + self.MSE(outputs[:,[1]], y_targets)).mean()

class UncertainLocLoss(nn.Module):
    """
    Uncertainty weighted loss for X and Y location components

    ...

    Attributes
    ----------
    log_vars : List[torch.tensor]
        list of log-transformed task-specific noise estimates
    MSE : nn.Module
        PyTorch MSE loss function with no reduction
    """

    def __init__(self, log_var_list=None):
        """
        Construct attributes

        log_vars : List[torch.tensor]
            list of log-transformed task-specific noise estimates
        MSE : nn.Module
            PyTorch MSE loss function with no reduction
        """

        super(UncertainLocLoss, self).__init__()
        self.MSE = nn.MSELoss(reduction='none')

        # Initialize log vars or resume
        if log_var_list is None:
            self.log_vars = [torch.tensor(0., requires_grad=True, dtype=torch.float32), torch.tensor(0., requires_grad=True, dtype=torch.float32)]
        else:
            self.log_vars = [torch.tensor(log_var_list[0].item(), requires_grad=True, dtype=torch.float32), torch.tensor(log_var_list[1].item(), requires_grad=True, dtype=torch.float32)]

    def forward(self, outputs, x_targets, y_targets):
        """
        Uncertainty weighted loss.
        """
        x_loss = self.MSE(outputs[:,[0]], x_targets)
        y_loss = self.MSE(outputs[:,[1]], y_targets)
        expx = torch.exp(-self.log_vars[0])
        expy = torch.exp(-self.log_vars[1])
        return (0.5*expx*x_loss + 0.5*expy*y_loss + 0.5*self.log_vars[0] + 0.5*self.log_vars[1]).mean()

class UncertaintyPredictionLocLoss(nn.Module):
    """
    Uncertainty loss based on independent normal distributions. The model should
    output a predicted mean and standard devaition which are used to parameterize
    the conditional distributions (per sample).
    """
    
    def __init__(self):
        super(UncertaintyPredictionLocLoss, self).__init__()

    def forward(self, outputs, x_targets, y_targets):
        """
        NLL aggregate loss parameterized by modle.
        """
        # extact parameters
        x_mu = outputs[:,[0]]
        x_std = torch.exp(outputs[:,[1]])
        y_mu = outputs[:,[2]]
        y_std = torch.exp(outputs[:,[3]])

        # instantiate distributions
        cond_dist_x = Normal(loc=x_mu, scale=x_std)
        cond_dist_y = Normal(loc=y_mu, scale=y_std)

        # sum NLL for both coordinates
        loss = -1*cond_dist_x.log_prob(x_targets) - cond_dist_y.log_prob(y_targets)

        return loss.mean()