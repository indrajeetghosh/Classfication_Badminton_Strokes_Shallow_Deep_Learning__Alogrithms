#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

# import keras
import pandas as pd
# import tensorflow as tf
import scipy.io as sio
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os
import h5py

from numpy import array, mean, std, dstack,argmax
from pandas import read_csv
from matplotlib import pyplot


# In[ ]:


warnings.filterwarnings('ignore')


# In[ ]:


def windows(series,window_size,overlap):
    start = 0
    while start+window_size <= series.shape[0]:
        yield int(start), int(start + window_size)
        start += window_size - overlap
        
        
def segment_signal(series,window_size,overlap):
    windowed_series = np.array([])
    for k in range(series.shape[1]):
        print("Windowing Column %d"%(k))
        temp_series = np.array([])
        for (start, end) in windows(series,window_size,overlap):
            if start==0:
                temp_series = series[start:end,k].T
            else:
                temp_series = np.vstack([temp_series,series[start:end,k].T])
        if k==0:
            windowed_series = temp_series
        else:
            windowed_series = np.vstack([windowed_series,temp_series])
    windowed_series = windowed_series.reshape((k+1,int(windowed_series.shape[0]/(k+1)),window_size))
    return windowed_series

def segment_label(label_series,window_size,overlap):
    windowed_labels = np.array([])
    for (start, end) in windows(label_series,window_size,overlap):

        if start==0:
            windowed_labels = max(label_series[start:end])[0]
        else:
            windowed_labels = np.vstack([windowed_labels,max(label_series[start:end])[0]])
    return windowed_labels    


# In[ ]:


result_dir = '...' 
filename = '..'
window_size = 32
overlap = 24


# In[ ]:


for file in os.listdir(result_dir):
    if not file.startswith('.'):
        save_dir = result_dir+file+'/'+filename
        print(save_dir)
        mat = sio.loadmat(save_dir)
        X_full_appended_temp = np.asarray(mat['Filename'])
        y_full_temp = np.asarray(mat['Filename']).T
        y_full_temp = np.squeeze(np.ndarray.tolist(y_full_temp))
        y_full_temp = y_full_temp.reshape(len(y_full_temp),1)
        X_full_appended = segment_signal(X_full_appended_temp,window_size,overlap)
        y_full = segment_label(y_full_temp,window_size,overlap)
         sio.savemat(save_dir+'Filename'%(window_size,overlap),{'Filename': X_full_appended, 
                                            'Filename':y_full,
                                            'window_size':window_size,
                                            'overlap':overlap,
                                           })
        print("Saved to "+ save_dir)        


# In[ ]:




