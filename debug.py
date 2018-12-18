# Beat tracking example
from __future__ import print_function

import os

import librosa as lbr
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from audio_processing import compute_mel

# Mel spectogram characteristic params
MAX_FREQ = 8000
WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MEL = 128

label_npy = ('labelTrain.npy', 'labelVal.npy', 'labelTest.npy')

def print_all_mel():
    search_path = os.getcwd()
    for dirpath, dirnames, filenames in os.walk(os.path.join(search_path, 'dataset')):
        for filename in filenames:
            mel_matrix = compute_mel(os.path.join(dirpath, filename))
            fig = plt.figure()
            lbr.display.specshow(lbr.power_to_db(mel_matrix, ref=np.max),
                                 y_axis='mel', fmax=MAX_FREQ,
                                 x_axis='time')
            plt.colorbar(format='%0.2f dB')
            y_value = os.path.basename(dirpath)
            if not os.path.exists(os.path.join('images', y_value)):
                os.mkdir(os.path.join('images', y_value))
            fig.savefig(os.path.join('images', y_value, os.path.splitext(filename)[0] + '.png'))
            plt.close('all')
            print(filename)


# Reads in the datasets from csv files, ignoring the first row and first col of the data
def read_dataset():
    label_train = np.load(label_npy[0])
    label_val = np.load(label_npy[1])
    label_test = np.load(label_npy[2])
    return label_train, label_val, label_test