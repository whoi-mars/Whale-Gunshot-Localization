from scipy.signal import stft
import yaml
import numpy as np

import torch
from torchvision import transforms

# load config file
with open("config.yaml", 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.Loader)

def to_spect(x):
    """
    Calculates spectrograms from timeseries. Note that this function assumes
    that the signal is in the last dimension of the input.

    Parameters
    ----------
    x : array-like
        Input timeseries'. Shape should be (# signal sets, ..., # samples/signal)
    
    Returns
    -------
    array-like
        Spectrograms of input timeseries'. The shape is (# signal sets OR 0, ..., 1, frequency, time)
    """

    # calculate STFTs
    [f, t, Z] = stft(x, 
                    fs=config['signal']['fs'], 
                    window=config['stft']['window'], 
                    nperseg=config['stft']['nperseg'], 
                    noverlap=config['stft']['noverlap'], 
                    nfft=config['stft']['nfft'],
                    axis=-1)

    # get dB power
    log_spect = 10*np.log10(np.abs(Z)**2)

    # get dimension where to flip the result and add channel dimension
    channel_dim = len(log_spect.shape) - 2

    # flip about time axis, and add channel dimension
    return np.expand_dims(np.flip(log_spect, axis=channel_dim), axis=channel_dim)

class Normalize1DChannel:
    """
    Normalizes spectrograms row-wise by mean-centering and dividing by std

    ...

    Attributes
    ----------
    mu_list : array-like
        list of row-wise means for each spectrogram
    std_list : array-like
        list of row-wise stds for each spectrogram
    num_signals : int
        number of input spectrograms
    """

    def __init__(self, mu_list, std_list):
        """
        Construct attributes.

        Parameters
        ----------
        mu_list : array-like
            list of row-wise means for each spectrogram
        std_list : array-like
            list of row-wise stds for each spectrogram
        """

        self.mu_list = torch.as_tensor(mu_list).float()
        self.std_list = torch.as_tensor(std_list).float()
        self.num_signals = mu_list.shape[0]

    def norm(self, x):
        """divide input spectrogram rows by provided mean and divide by provided std"""

        return (x - self.mu_list.view(self.num_signals, 1, -1, 1)) / (self.std_list.view(self.num_signals, 1, -1, 1))

    def __call__(self, tensor):
        return self.norm(tensor)

def get_image_transform():
    """
    Gets dictionary of spectrogram preprocessing transforms for training and evaluation.

    Returns
    -------
    dict
        dictionary with training and evaluation preprocessing transforms
    """

    # load mean and std
    mu_list = np.load(config['dataset']['data_directory'] + '/mean.npy', allow_pickle=True)
    std_list = np.load(config['dataset']['data_directory'] + '/std.npy', allow_pickle=True)
    
    # evaluation transforms
    transform_eval = transforms.Compose([
        Normalize1DChannel(mu_list, std_list)
    ])

    # training transforms
    transform_train = transforms.Compose([
        Normalize1DChannel(mu_list, std_list)
    ])

    return {'train' : transform_train, 'eval' : transform_eval}