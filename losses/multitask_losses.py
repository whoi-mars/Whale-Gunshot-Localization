import torch
import torch.nn as nn

class UncertainSelectiveMSEAndClass(nn.Module):
    
    """
    Class for loss which adds cross entropy loss and MSE loss
    for range predictions. The MSE loss only penalizes incorrect range
    predictions for examples of class 1 (with a call in the spectrogram).
    This loss also implements the approach from Kendall et al. (https://arxiv.org/abs/1705.07115) 
    to learn weights for the classification and ranging tasks.
    Parameters
    ----------
    log_var_list: List[torch.tensor], current log variables for resuming training. If 'None', they
                  will be initialized to zero.
    
    Returns
    -------
    r_c_loss_m: torch.tensor, total loss averaged over the batch.
    r_loss_m: torch.tensor, total range loss averaged over the batch.
    c_loss_m: torch.tensor, total class loss averaged over the batch.
    """
    
    def __init__(self, log_var_list=None):
        super().__init__()
        self.MSE = nn.MSELoss(reduction='none')

        # Learned variables for weighting the tasks
        if log_var_list is None:
            self.log_vars = [torch.tensor(0., requires_grad=True), torch.tensor(0., requires_grad=True)]
        else:
            self.log_vars = [torch.tensor(log_var_list[0].item(), dtype=torch.float32, requires_grad=True), torch.tensor(log_var_list[1].item(), dtype=torch.float32, requires_grad=True)]

    def forward(self, outputs, r_labels, c_labels):

        # Isolate ranges for call-containing example only
        call_outs = outputs[c_labels.squeeze() == 1, 0]
        call_r_labels = r_labels[c_labels.squeeze() == 1]

        # Get classification loss
        c_loss = torch.exp(-self.log_vars[1])*outputs[torch.arange(len(outputs),dtype=torch.long), (c_labels.squeeze()+1).long()] - torch.log(torch.sum(torch.exp(torch.exp(-self.log_vars[1])*outputs[:,1:]),dim=1))

        # Calculate range loss
        r_loss = self.MSE(call_outs.squeeze(), call_r_labels.squeeze())
        r_precision = torch.exp(-self.log_vars[0])
        r_loss = 0.5*(r_precision*r_loss + self.log_vars[0])
        r_loss_final = torch.zeros_like(c_loss) 
        r_loss_final[c_labels.squeeze() == 1] = r_loss

        # Get means
        r_loss_m = r_loss_final.mean()
        c_loss_m = (-1*c_loss).mean()
        r_c_loss_m = (r_loss_final - c_loss).mean()
        
        return r_c_loss_m, r_loss_m, c_loss_m