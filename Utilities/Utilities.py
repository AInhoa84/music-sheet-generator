#!/usr/bin/env python
# coding: utf-8

# In[1]:


def create_xy(df, target_column):
    return df.drop(target_column, axis=1), df[target_column]

def split_data(df, target_column):
    X_train, y_train = create_xy(df.sample(round(0.8*df.shape[0])), target_column)
    df = df.drop(X_train.index)
    X_val, y_val = create_xy(df.sample(round(0.5*df.shape[0])), target_column)
    df = df.drop(X_val.index)
    X_test, y_test = create_xy(df, target_column)
    return X_train, y_train, X_val, y_val, X_test, y_test


# In[2]:


def envelope(y, n):
    env_pos = []
    env_neg = []

    for i in range(0, len(y), n):
        env_pos += n * [np.max(y[i:(n+i)])]
        env_neg += n * [np.min(y[i:(n+i)])]
        
    return env_pos, env_neg

def frontiers(y, env, k):
    previous = np.array(env)[:-1]
    next = np.array(env)[1:]
    front = np.argwhere((next >= k*previous) & (next > 0.025)).flatten()
    front = np.append(front, len(y[::-1][np.argwhere(y[::-1] >= 0.005)[0][0]:]))
    
    return front


# In[3]:


import json
from keras.models import model_from_json

def load_NN(name):
    with open(name + "_NN_architecture.json", 'r') as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(name + "_NN_weights.h5")
    model.summary()
    return model


# In[4]:


def round_to_base(x, base):
    return base * round(x/base)


# In[14]:


import librosa

def load_file(file):
    y, sr = librosa.load(file, mono=False)
    y = librosa.core.to_mono(y)
    y = librosa.resample(y, sr, 8000)
    return y

def temp_data(y, samples, norm):
    if norm:
        data = pd.DataFrame({'x{}'.format(j): [y[j]/np.max(y)] for j in range(samples)})
    else:
        data = pd.DataFrame({'x{}'.format(j): [y[j]] for j in range(samples)})
    return data

def spectral_data(y, samples, norm):
    if norm:
        w = abs(np.fft.fft(y, n=samples*2))
        freqs = np.fft.fftfreq(len(w))
        data = pd.DataFrame({"x{}".format(j): [w[freqs >= 0][j]/max(w)] for j in range(samples)})
    else:
        w = abs(np.fft.fft(y, n=samples*2))
        freqs = np.fft.fftfreq(len(w))
        data = pd.DataFrame({"x{}".format(j): [w[freqs >= 0][j]] for j in range(samples)})


# In[13]:


from tqdm import tqdm_notebook as tqdm

def dir_to_data(directory, function, *args):
    data = pd.DataFrame()
    pbar = tqdm(os.listdir(directory))
    
    for file in pbar:
        pbar.set_description("Processing %s" % file)
        df = function(directory + file, *args)
        data = data.append(df)
        
    data = data.reset_index().drop("index", axis=1)
    return data


# In[15]:


def apply_window(y, size, disp, function, *args):
    results = []
    for i in range(0, len(y), disp):
        window = y[i:i+size]
        results.append(function(window, *args))
    return results

