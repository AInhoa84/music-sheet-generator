import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

class waveform:
    def __init__(self, wave, rs=8000):
        """
        Loads an audio file into a waveform object
        
        """
        # Check whether wave is a file name or an array
        if type(wave) == str:
            y, sr = librosa.load(wave, mono=False)
            y = librosa.core.to_mono(y)
            y = librosa.resample(y, sr, rs)
            self.y = y
        else:
            self.y = wave
        
    def envelope(self, n):
        """
        Calculates the positive and negative envelopes of a wave
        
        """
        env_pos = []
        env_neg = []

        for i in range(0, len(self.y), n):
            env_pos += n * [np.max(self.y[i:(n+i)])]
            env_neg += n * [np.min(self.y[i:(n+i)])]

        return env_pos, env_neg
    
    def temp_data(self, samples, norm=False):
        """
        Extracts temporal data from a waveform

        Args:
            samples (int): Number of samples to get from the waveform
            norm (bool): True to perform normalization

        Returns:
            DataFrame: Extracted data
        """
        if norm:
            data = pd.DataFrame({'x{}'.format(j): [self.y[j]/np.max(self.y)] for j in range(samples)})
        else:
            data = pd.DataFrame({'x{}'.format(j): [self.y[j]] for j in range(samples)})
        return data
    
    def spectral_data(self, samples, norm=False):
        """
        Extracts spectral datafrom a waveform using FFT

        Args:
            samples (int): Number of samples to get from the waveform
            norm (bool): True to perform normalization

        Returns:
            DataFrame: Extracted data
        """
        if norm:
            w = abs(np.fft.fft(self.y, n=samples*2))
            freqs = np.fft.fftfreq(len(w))
            data = pd.DataFrame({"x{}".format(j): [w[freqs >= 0][j]/max(w)] for j in range(samples)})
        else:
            w = abs(np.fft.fft(self.y, n=samples*2))
            freqs = np.fft.fftfreq(len(w))
            data = pd.DataFrame({"x{}".format(j): [w[freqs >= 0][j]] for j in range(samples)})
        return data
    
    def apply_window(self, size, disp, function, convert=False, temp=True, norm=False, *args):
        results = []
        for i in tqdm(range(0, len(self.y)-size, disp), leave = False):
            window = self.y[i:i+size]
            if temp and convert:
                window = temp_data(window, size, norm)
            elif (not temp) and convert:
                window = spectral_data(window, size, norm)
            results.append([i, function(window, *args)])
        return results


# Data utilities
def create_xy(df, target_column):
    """
    Separates features and target
    
    Args:
        df (DataFrame): Original dataframe
        target_column (str): Name of the target column
    
    Returns:
        Dataframe: Feature dataframe
        Dataframe: Target dataframe
    
    """
    return df.drop(target_column, axis=1), df[target_column]

def split_data(df, target_column):
    """
    Splits data into test, train and validation
    
    Args:
        df (DataFrame): Original dataframe
        target_column (str): Name of the target column
    
    Returns:
        DataFrame: Train feature dataframe
        DataFrame: Train target dataframe
        DataFrame: Validation feature dataframe
        DataFrame: Validation target dataframe
        DataFrame: Test feature dataframe
        DataFrame: Test target dataframe
    """
    X_train, y_train = create_xy(df.sample(round(0.8*df.shape[0])), target_column)
    df = df.drop(X_train.index)
    X_val, y_val = create_xy(df.sample(round(0.5*df.shape[0])), target_column)
    df = df.drop(X_val.index)
    X_test, y_test = create_xy(df, target_column)
    return X_train, y_train, X_val, y_val, X_test, y_test


# Convert audio files from a directory into data
from tqdm import tqdm_notebook as tqdm

def dir_to_data(directory, function, *args):
    """
    Applies a function to every file in a directory
    
    Args:
        directory (str): Name of the directory
        function : Function that will be applied to each file
        
    Returns:
        DataFrame: Extracted data
    """
    data = pd.DataFrame()
    pbar = tqdm(os.listdir(directory))
    
    for file in pbar:
        pbar.set_description("Processing %s" % file)
        df = function(directory + file, *args)
        data = data.append(df)
        
    data = data.reset_index().drop("index", axis=1)
    return data


# ## Included in waveform class
def envelope(y, n):
    env_pos = []
    env_neg = []

    for i in range(0, len(y), n):
        env_pos += n * [np.max(y[i:(n+i)])]
        env_neg += n * [np.min(y[i:(n+i)])]
        
    return env_pos, env_neg

def apply_window(y, size, disp, function, convert=False, temp=True, norm=False, *args):
    results = []
    for i in tqdm(range(0, len(y)-size, disp), leave = False):
        window = y[i:i+size]
        if temp and convert:
            window = temp_data(window, size, norm)
        elif (not temp) and convert:
            window = spectral_data(window, size, norm)
        results.append([i, function(window, *args)])
    return results

def load_file(file, rs):
    """
    Loads an audio file into an array
    
    Args:
        file (str): File name
        
    Returns:
        Array: Waveform
    """
    y, sr = librosa.load(file, mono=False)
    y = librosa.core.to_mono(y)
    y = librosa.resample(y, sr, rs)
    return y

def temp_data(y, samples, norm):
    """
    Extracts temporal data from a waveform
    
    Args:
        y (Array): Waveform
        samples (int): Number of samples to get from the waveform
        norm (bool): True to perform normalization
        
    Returns:
        DataFrame: Extracted data
    """
    if norm:
        data = pd.DataFrame({'x{}'.format(j): [y[j]/np.max(y)] for j in range(samples)})
    else:
        data = pd.DataFrame({'x{}'.format(j): [y[j]] for j in range(samples)})
    return data

def spectral_data(y, samples, norm):
    """
    Extracts spectral data from a waveform
    
    Args:
        y (Array): Waveform
        samples (int): Number of samples to get from the waveform
        norm (bool): True to perform normalization
        
    Returns:
        DataFrame: Extracted data
    """
    if norm:
        w = abs(np.fft.fft(y, n=samples*2))
        freqs = np.fft.fftfreq(len(w))
        data = pd.DataFrame({"x{}".format(j): [w[freqs >= 0][j]/max(w)] for j in range(samples)})
    else:
        w = abs(np.fft.fft(y, n=samples*2))
        freqs = np.fft.fftfreq(len(w))
        data = pd.DataFrame({"x{}".format(j): [w[freqs >= 0][j]] for j in range(samples)})
    return data

def frontiers(y, env, k, use_desc=False):
    previous = np.array(env)[:-1]
    current = np.array(env)[1:]
    if use_desc:
        front = np.argwhere(((current >= k*previous) | (previous >= k*current)) & (current > 0.025)).flatten()
    else:
        front = np.argwhere((current >= k*previous) & (current > 0.025)).flatten()
        
    try:
        front = np.append(front, len(y[::-1][np.argwhere(y[::-1] >= 0.005)[0][0]:]))
    except:
        pass
    
    return front

import json
from keras.models import model_from_json

def load_NN(name, verbose=True):
    with open(name + "_NN_architecture.json", 'r') as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(name + "_NN_weights.h5")
    if verbose:
        model.summary()
    return model

def round_to_base(x, base):
    x = np.array(x)
    return base * np.round(x/base)

