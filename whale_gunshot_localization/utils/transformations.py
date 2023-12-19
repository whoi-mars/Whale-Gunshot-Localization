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
        self.num_signals = self.mu_list.shape[0]

    def norm(self, x):
        """
        Divide input spectrogram rows by provided mean and divide by provided std.
        This assumes the input is of shape (# spectrograms, 1, # frequency bins, # time bins).
        """
        
        return (x - self.mu_list.view(self.num_signals, 1, -1, 1)) / (self.std_list.view(self.num_signals, 1, -1, 1))

    def __call__(self, tensor):
        return self.norm(tensor)

class BlockTOSSIT:
    """
    Augmentation to zero-out entire TOSSITs randomly. There will always be
    'min_received' TOSSITs, and combinations of remaining TOSSITs to be zeroed
    will be applied uniformly.

    ...

    Attributes
    ----------
    min_received : int
        minimum number of unblocked TOSSITs
    """
    def __init__(self, min_received):
        """
        Construct attributes.

        Parameters
        ----------
        min_received : int
            minimum number of unblocked TOSSITs
        """
        self.min_received = min_received
    
    def _uniform_ones_bitstring(self, max_ones):
        """
        Returns a bit string of length 'max_ones' with uniformly distributed
        number of ones.

        Parameters
        ----------
        max_ones : 
            length of the bitstring to return
        
        Returns
        -------
        torch.Tensor : torch.bool
            bit string
        """
        res = torch.zeros((max_ones,), dtype=torch.bool)
        
        num_ones = torch.randint(high=max_ones+1, size=())
        idx = torch.randperm(max_ones)[:num_ones]
        res[idx] = 1

        return res

    def block(self, x):
        """
        Zero out some TOSSITs uniformly while maintaining 'min_received'.

        Parameters
        ----------
        x : array-like
            TOSSIT signals

        Returns
        -------
        x : array-like
            TOSSIT signals with some zeroed out
        """

        # get total number of TOSSITs and number to decide to keep or not
        num_TOSSITs = x.shape[0]
        remainder = num_TOSSITs - self.min_received

        # logical array for final output
        bidx = torch.zeros((num_TOSSITs,), dtype=torch.bool)
        
        # create indexing arrays for min_received and random deletion
        c_baseline = torch.ones((num_TOSSITs,), dtype=torch.bool)
        baseline = torch.randperm(num_TOSSITs)[:self.min_received]
        c_baseline[baseline] = 0

        #execute
        bidx[baseline] = 0
        bidx[c_baseline] = self._uniform_ones_bitstring(remainder)

        # apply
        x[bidx] = 0

        return x

    def __call__(self, tensor):
        return self.block(tensor)

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

def get_image_transform(stats=False):
    """
    Gets dictionary of spectrogram preprocessing transforms for training and evaluation.

    Parameters
    ----------
    stats : bool
        whether or not to load the stats transforms exclusively

    Returns
    -------
    dict
        dictionary with training and evaluation preprocessing transforms
    """

    if stats:
        # transforms for calculating stats
        transform_stats = transforms.Compose([
            BlockTOSSIT(3),
        ])
        return {'stats' : transform_stats}

    # load mean and std
    mu_list = np.load(config['dataset']['data_directory'] + '/mean.npy', allow_pickle=True)
    std_list = np.load(config['dataset']['data_directory'] + '/std.npy', allow_pickle=True)
    
    # evaluation transforms
    transform_eval = transforms.Compose([
        Normalize1DChannel(mu_list, std_list)
    ])

    # training transforms
    transform_train = transforms.Compose([
        Normalize1DChannel(mu_list, std_list),
        BlockTOSSIT(3),
    ])

    return {'train' : transform_train, 'eval' : transform_eval}

def get_image_transform_range_classify():
    """
    Gets dictionary of spectrogram preprocessing transforms for training and evaluation.

    Returns
    -------
    dict
        dictionary with training and evaluation preprocessing transforms
    """

    # load mean and std
    mu_list = np.load(config['dataset']['data_directory'] + '/mean_range_classification.npy', allow_pickle=True)
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