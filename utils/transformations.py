from scipy.signal import stft
import yaml
import numpy as np

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