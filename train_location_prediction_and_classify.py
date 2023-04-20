import os
import argparse
import copy
import time
import math

from tqdm import tqdm
import numpy as np
import yaml
import wandb
import torch
import torch.nn.functional as F

from datasets.dataloaders import get_dataloaders_range_classify
from utils.transformations import get_image_transform_range_classify
from models.tcn_archs import TCNRangeAndClassify
from losses.multitask_losses import UncertainSelectiveMSEAndClass

# load config file
with open("config.yaml", 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.Loader)

# training arguments
parser = argparse.ArgumentParser(description="Training on simulated data from adiabatic modes simulator")
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size (default: 128)')
parser.add_argument('--start_epoch', type=int, default=1,
                    help='epoch number to start at, inclusive (default: 1)')
parser.add_argument('--end_epoch', type=int, default=10,
                    help='epoch number to end at, inclusive (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--channels', type=int, nargs='+', default=7*[368],
                    help='number of TCN blocks including (fusion default: [250] + 7*[500])')
parser.add_argument('--kernel_size', type=int, default=6,
                    help='size of 1D kernel (default: 6)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='spatial dropout parameter (default: 0.2)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay (default: 0.0)')
parser.add_argument('--save_all_epochs', action='store_true',
                    help='store weights for all epochs during training (default: False)')
parser.add_argument('--DP', action='store_true',
                    help='use PyTorch nn.DataParallel')
parser.add_argument('--checkpoint_dir', type=str, default='model',
                    help='directory where model weight checkpoints will be saved (default: model)')
parser.add_argument('--verbose', action='store_true',
                    help='display progress while training (default: False)')
parser.add_argument('--wb_id', type=str, nargs=None,
                    help='id of weights and biases run to continue (default: None)')
parser.add_argument('--no_wb', action='store_true',
                    help='disable weights and biases logging (default: False)')

# parse training arguments
args = parser.parse_args()

# setup weights and biases
if not args.no_wb:
    # login
    wandb.login()

    # get epochs currently trained for if resuming
    if args.wb_id is not None:
        api = wandb.Api()
        run = api.run("markg98/simultaneous-range-classify/" + args.wb_id)
        epoch_offset = run.config['epochs']
    else:
        epoch_offset = 0
    
    # intialize
    wandb.init(
        project="simultaneous-range-classify",
        config={
            "epochs" : args.end_epoch - args.start_epoch + 1 + epoch_offset,
            "batch_size" : args.batch_size,
            "lr" : args.lr,
            "window" : config['stft']['window'],
            "nperseg" : config['stft']['nperseg'],
            "noverlap" : config['stft']['noverlap'],
            "nfft" : config['stft']['nfft'],
            "max_x" : config['scaling']['max_x'],
            "max_y" : config['scaling']['max_y'],
            "max_r" : config['scaling']['max_r'],
            "checkpoint_directory" : args.checkpoint_dir,
            "num_channels" : args.channels,
            "dropout" : args.dropout,
            "weight_decay" : args.weight_decay,
            "clip" : args.clip,
        },
        id=args.wb_id,
        resume= True if args.wb_id is not None else False
    )

# get devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("\nUsing the GPU!")
    # set GPU-related seeds
    torch.cuda.manual_seed(args.seed)
else:
    print("WARNING: Could not find GPU. Using CPU only.")

# set PyTorch/numpy seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# get dataloaders
dl = get_dataloaders_range_classify(splits=['train', 'val'],
                                    batch_size=args.batch_size,
                                    drop_last=False,
                                    shuffle=True,
                                    transform=get_image_transform_range_classify(),
                                    squeeze=True,
                                    num_workers=20,
                                    pin_memory=True)
n_steps_per_epoch = len(dl['train'])

# get label scaling constants for error calculations
max_r = dl['train'].dataset.max_r

# print size of input
print(f"spectrogram size: {dl['train'].dataset.size}\n")

# setup model checkpoint directory
save_dir = os.path.join(config['models']['checkpoints_directories'], args.checkpoint_dir)
os.makedirs(save_dir, exist_ok=True)

# initialize TCN model
model = TCNRangeAndClassify(input_size=dl['train'].dataset.size[0], 
                            num_channels=args.channels,
                            kernel_size=args.kernel_size,
                            dropout=args.dropout).to(device)

# load model/optimizer checkpoint or start from scratch
if args.start_epoch > 1:
    # try to load previous epoch save
    try:
        checkpoint = torch.load(os.path.join(save_dir, f'weights_{args.start_epoch - 1}.pt'))
    except:
        raise ValueError("Desired start epoch does not have a corresponding set of saved model weights.")
    # load states if possible
    model.load_state_dict(checkpoint['model_state_dict'])
    log_var_list = checkpoint['log_vars']
    # set loss
    criterion = UncertainSelectiveMSEAndClass(log_var_list=log_var_list).to(device)
    # initialize optimizer
    optimizer = torch.optim.Adam([p for p in model.parameters()] + [lv for lv in criterion.log_vars], lr=args.lr, weight_decay=args.weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    log_var_list = None
    # set loss
    criterion = UncertainSelectiveMSEAndClass(log_var_list=log_var_list).to(device)
    # initialize optimizer
    optimizer = torch.optim.Adam([p for p in model.parameters()] + [lv for lv in criterion.log_vars], lr=args.lr, weight_decay=args.weight_decay)

# data parallel
if args.DP:
    model = torch.nn.DataParallel(model)

def get_model_state_dict(model):

    """
    Save the model state dictionary, taking into account whether
    or not the model is wrapped in a torch.nn.DataParallel object.
    
    Parameters
    ----------
    model: torch.nn.Module
        model to save state dict for.
    
    Returns
    -------
    state_dict 
        state dictionary for model.
    """

    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    else:
        return model.state_dict()

# def scale_to_km(x, dim):

#     if dim == 'x':
#         return (x*(max_x - min_x) + min_x) / 1000
#     elif dim == 'y':
#         return (x*(max_y - min_y) + min_y) / 1000
#     else:
#         raise ValueError('invalid input')

def train(model, dataloaders, criterion, optimizer, end_epoch=args.end_epoch, save_dir=save_dir, save_all_epochs=args.save_all_epochs, start_epoch=args.start_epoch, verbose=args.verbose):
    """
    Training function.

    Parameters
    ----------
    model : nn.Module
        model to be trained
    dataloaders : dict
        dictionary of dataloaders for each split
    criterion : nn.Module
        loss function
    optimizer : optimizer from torch.optim
        chosen optimizer
    end_epoch : int
        epoch to end training on, inclusive
    save_dir : str
        path to directories where all model checkpoints are stored
    save_all_epochs : bool
        save model weights after each epoch
    start_epoch : int
        epoch to start training on, inclusive
    verbose : bool
        display progress while training
    """
    # time training
    since = time.time()

    # initialize best model
    best_model_wts = copy.deepcopy(get_model_state_dict(model))
    best_opt_state = copy.deepcopy(optimizer.state_dict())
    best_log_vars = copy.deepcopy(criterion.log_vars)
    best_mse = float('inf')

    # train/val loops
    for epoch in range(start_epoch, end_epoch+1):
        print(f"Epoch {epoch}/{end_epoch}")
        print('-'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # keep track of loss, and error
            running_loss = 0.0
            running_loss_r = 0.0
            running_loss_c = 0.0
            running_r_sq_error = 0.0
            running_corrects = 0
            running_example_count = 0
            running_call_count = 0

            # process batches
            for step, (inputs, targets_c, targets_r) in enumerate(tqdm(dataloaders[phase], disable=not verbose)):

                # put data/labels on device
                inputs = inputs.to(device)
                targets_c = targets_c.to(device)
                targets_r = targets_r.to(device)

                # zero out gradient for new batch
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # get model outputs and loss
                    outputs = model(inputs)
                    loss, r_loss, c_loss = criterion(outputs, targets_r, targets_c)
                    # backpropogate
                    if phase == 'train':
                        if args.clip > 0:
                            torch.nn.clip_grad_norm_([p for p in model.parameters()] + [lv for lv in criterion.log_vars], args.clip)
                        loss.backward()
                        optimizer.step()

                # get running loss sum and running sqared error sum (in km) for the X and Y components of location
                running_loss += loss.item()*inputs.size(0)
                running_loss_r += r_loss.item()*inputs.size(0)
                running_loss_c += c_loss.item()*inputs.size(0)

                # get running square error for whole epoch
                outputs_r_TP = outputs[torch.where((F.softmax(outputs[:,1:],dim=1)[:,1].squeeze() >= 0.5) & (targets_c.squeeze() == 1))[0],0].detach()
                targets_r_TP = targets_r[torch.where((F.softmax(outputs[:,1:],dim=1)[:,1].squeeze() >= 0.5) & (targets_c.squeeze() == 1))].detach()
                r_mse = F.mse_loss(outputs_r_TP * max_r / 1000, targets_r_TP.squeeze() * max_r / 1000, reduction='none')
                running_r_sq_error += r_mse.sum()

                # running class
                running_example_count += len(targets_c.detach())
                running_call_count += targets_c.sum().item()

                running_corrects += ((F.softmax(outputs[:,1:].detach(),dim=1)[:,1].squeeze() >= 0.5) == targets_c.detach().squeeze()).sum()
            
                if (not args.no_wb) and phase == 'train':
                    step_metrics = {"train/train_loss" : loss,
                                    "train/class_loss" : c_loss,
                                    "train/range_loss" : r_loss,
                                    "train/epoch" : (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                                    "train/LVr" : criterion.log_vars[0],
                                    "train/LVc" : criterion.log_vars[1],
                                    "train/r_sq_error" : r_mse.mean(),}
                    if step + 1 < n_steps_per_epoch:
                        wandb.log(step_metrics)

            # calculate epoch statistics
            epoch_loss = running_loss / dataloaders[phase].batch_sampler.num_samples
            epoch_loss_r = running_loss_r / running_call_count
            epoch_loss_c = running_loss_c / dataloaders[phase].batch_sampler.num_samples
            epoch_r_mse = running_r_sq_error / running_call_count

            # end of epoch wandb logging
            if not args.no_wb:
                if phase == 'train':
                    train_metrics = {"train/train_avg_loss" : epoch_loss,
                                     "train/train_avg_loss_c" : epoch_loss_c,
                                     "train/train_avg_loss_r" : epoch_loss_r,
                                     "train/train_r_rmse" : torch.sqrt(epoch_r_mse),
                                     "train/acc" : running_corrects / running_example_count,}
                    wandb.log({**step_metrics, **train_metrics})
                else:
                    val_metrics = {"val/val_avg_loss" : epoch_loss,
                                   "val/val_avg_loss_c" : epoch_loss_c,
                                   "val/val_avg_loss_r" : epoch_loss_r,
                                   "val/val_r_rmse" : torch.sqrt(epoch_r_mse),
                                   "val/acc" : running_corrects / running_example_count,}
                    wandb.log(val_metrics)

            # print epoch information
            print("{} Loss: {:.4f} -- RMSE: {:.4f} km -- ACC: {:.4F} -- LVr: {:.4f} -- LVc: {:.4f}".format(phase, epoch_loss, torch.sqrt(epoch_r_mse), running_corrects / running_example_count,criterion.log_vars[0], criterion.log_vars[1]))

            # check if we update best model
            if phase == 'val' and epoch_r_mse < best_mse:
                best_mse = epoch_r_mse
                best_model_wts = copy.deepcopy(get_model_state_dict(model))
                best_opt_state = copy.deepcopy(optimizer.state_dict())
                best_log_vars = copy.deepcopy(criterion.log_vars)
            
            # save model weights if saving all epochs
            if phase == 'train' and save_all_epochs:
                torch.save({"model_state_dict" : get_model_state_dict(model),
                            "optimizer_state_dict" : optimizer.state_dict(),
                            "log_vars" : criterion.log_vars},
                            os.path.join(save_dir, f'weights_{epoch}.pt'))

        # new line after train/val cycle
        print()

    # training done!
    time_elapsed = time.time() - since
    print("Training completed in {:.0f}h {:.0f}m {:.0f}s".format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))
    print("Best avg MSE: {:.4f} km^2".format(best_mse))
    print("Best avg RMSE: {:.4f} km".format(torch.sqrt(best_mse)))

    # save best model weights
    torch.save({"model_state_dict" : best_model_wts,
                "optimizer_state_dict" : best_opt_state,
                "log_vars" : best_log_vars},
                os.path.join(save_dir, f'weights_best.pt'))

    # if we're not saving every epoch, save the last one
    if not save_all_epochs:
        torch.save({"model_state_dict" : get_model_state_dict(model),
                    "optimizer_state_dict" : optimizer.state_dict(),
                    "log_vars" : criterion.log_vars},
                    os.path.join(save_dir, f'weights_{epoch}.pt'))

if __name__ == "__main__":
    train(model=model,
        dataloaders=dl,
        criterion=criterion,
        optimizer=optimizer)