from scipy.signal import stft
import yaml
import numpy as np

import torch
from torchvision import transforms

# load config file
from whale_gunshot_localization import config

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
    [_, _, Z] = stft(x, 
                    fs=config['signal']['fs'], 
                    window=config['stft']['window'], 
                    nperseg=config['stft']['nperseg'], 
                    noverlap=config['stft']['noverlap'], 
                    nfft=config['stft']['nfft'],
                    axis=-1)

    # get dB power
    log_spect = 10*np.log10(np.abs(Z) ** 2)
    
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

    def norm(self, x):
        """
        Divide input spectrogram rows by provided mean and divide by provided std.
        This assumes the input is of shape (..., # frequency bins, # time bins).
        """
        # print("x", x.shape)
        shape = (*[1 for i in range(len(x.shape) - 2)], -1, 1)
        # print("shape", shape)
        return (x - self.mu_list.view(*shape)) / (self.std_list.view(*shape))

    def __call__(self, tensor):
        return self.norm(tensor)

class FrequencyBandZeroing:

    """
    Class to be used in transforms.Compose() to randomly zero out frequencies.

    ...

    Attributes
    ----------
    max_freq_width : int
        maximum width of continuous frequency band to zero out
    max_t_width : int
        maximum width of continuous time band to zero out
    num_f : int
        how many continuous frequency bands to zero out
    num_t : int 
        how many continuous time bands to zero out
    """

    def  __init__(self, max_freq_width = 30, max_t_width=30, num_f=2, num_t=4):
        """
        Construct attributes

        Parameters
        ----------
        max_freq_width : int
            maximum width of continuous frequency band to zero out
        max_t_width : int
            maximum width of continuous time band to zero out
        num_f : int
            how many continuous frequency bands to zero out
        num_t : int 
            how many continuous time bands to zero out
        """
        
        self.max_freq_width = max_freq_width
        self.max_t_width = max_t_width
        self.num_f = num_f
        self.num_t = num_t

    def freq_band_zeroing(self, x):

        """
        Function to randomly zero-out a band of frequencies and/or times

        Parameters
        ----------
        x : array-like 
            input spectrogram
        max_freq_width : int
            maximum continuous bandwidth to zero

        Returns
        -------
        x : array-like
            spectrogram with zeroed-out frequencies
        """

        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            N, _, H, W = x.shape
        elif len(x.shape) == 4:
            N, _, H, W = x.shape
        else:
            raise ValueError(f"x must be shape (N, H, W) or (N, C, H, W), is now {len(x.shape)}")
        
        zero_width_f = torch.randint(0, self.max_freq_width, size=(self.num_f,))
        offset_f = torch.randint(0, H - zero_width_f.max(), size=(self.num_f,))

        for wf, of in zip(zero_width_f, offset_f):
            x[:,:,of:(of + wf),:] = 0

        if self.num_t:
            zero_width_t = torch.randint(0, self.max_t_width, size=(self.num_t,))
            offset_t = torch.randint(0, W - zero_width_t.max(), size=(self.num_t,))
            
            for wf, of in zip(zero_width_t, offset_t):
                x[:,:,:,of:(of + wf)] = 0

        return x

    def __call__(self, tensor):
        return self.freq_band_zeroing(tensor)

def random_wrap(x):
    """
    Randomly wrap a time-domain signal 
    
    Parameters
    ----------
    x : array-like
        time-domain signal
    
    Returns
    -------
    array-like
        randomly wrapped time-domain signal
    """

    # get max shift possible
    max_shifts = len(x)

    shifts = np.c_[np.random.randint(low=0, high=max_shifts, size=x.shape[0])]
    return x[np.c_[:x.shape[0]], (np.r_[:x.shape[1]] - shifts) % x.shape[1]]

    # shift
    # return np.roll(x, shift=np.random.randint(low=0, high=max_shifts))


def get_image_transform_range_classify(mu_list=None, std_list=None):
    """
    Gets dictionary of spectrogram preprocessing transforms for training and evaluation.

    Returns
    -------
    dict
        dictionary with training and evaluation preprocessing transforms
    """

    # load mean and std
    if mu_list is None:
        mu_list = np.load(config['dataset']['data_directory'] + '/mean_range_classification.npy', allow_pickle=True)
    if std_list is None:
        std_list = np.load(config['dataset']['data_directory'] + '/std_range_classification.npy', allow_pickle=True)

    # evaluation transforms
    transform_eval = transforms.Compose([
        Normalize1DChannel(mu_list, std_list),
    ])

    # training transforms
    transform_train = transforms.Compose([
        Normalize1DChannel(mu_list, std_list),
        FrequencyBandZeroing(max_freq_width=30, num_t=0),
    ])

    return {'train' : transform_train, 'eval' : transform_eval}