import torch
import torch.nn as nn

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

 