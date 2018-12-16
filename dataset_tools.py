from __future__ import print_function
from audio_processing import compute_STFT
from audio_processing import create_grey_spectogram
from audio_processing import display_spectogram
import numpy as np
import os
import pandas as pd

# Default sampling parameters
sampling_period = 30.0
sampling_rate = 44100      # in Hz

# Default spectogram parameters
image_size = 200.0

# Default folder parameters
folder = 'dataset'

# Dataset class range
genre = np.asarray(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
                    'pop', 'reggae', 'rock'])

# Default train:val:test ratio which adds up to 1
split_ratio = (0.6, 0.2, 0.2)

# Default data storage names
x_csv = ('trainX.csv', 'valX.csv', 'testX.csv')
y_csv = ('trainY.csv', 'valY.csv', 'testY.csv')


def prepare_dataset(path):
    search_path = os.path.join(path, folder)
    print('============ Attempting to start the conversion...')
    x_combined, y_combined = create_dataset(search_path)
    print('============ Finished. Attempting to split dataset...')
    x_split, y_split = split_dataset(ratios=split_ratio, X=x_combined, Y=y_combined,
                                                                   total=x_combined.shape[0])
    print('============ Finished. Attempting to save dataset to disk...')
    save_dataset(x_split, y_split)


def create_dataset(search_path):
    dataset_x = []
    dataset_y = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(search_path):
        for filename in filenames:
            X = create_X(os.path.join(dirpath, filename))
            y_value = os.path.basename(dirpath)
            Y = create_Y(y_value)
            dataset_x.append(X)
            dataset_y.append(Y)
            count += 1
            print('.', end='')
        print(' {}%'.format((count / 1000.0) * 100))
    dataset_x = np.vstack(dataset_x)
    dataset_y = np.vstack(dataset_y)
    return dataset_x, dataset_y


def split_dataset(ratios, X, Y, total):
    x = []
    y = []
    end = 0
    for ratio in ratios:
        start = end
        end = start + int(ratio * total)
        x.append(X[start:end])
        y.append(Y[start:end])
    return x, y


def save_dataset(x_split, y_split):
    i = 0
    while i < len(x_split):
        df = pd.DataFrame(x_split[i])
        df.to_csv(x_csv[i])
        print('saving to {}'.format(x_csv[i]))
        i += 1
    j = 0
    while j < len(y_split):
        df = pd.DataFrame(y_split[j])
        df.to_csv(y_csv[j])
        print('saving to {}'.format(y_csv[j]))
        j += 1


# Reads in the datasets from csv files, ignoring the first row and first col of the data
def read_dataset():
    dfx_train = pd.read_csv(x_csv[0], index_col=0)
    x_train = __reshape(dfx_train.values, rows=int(image_size), cols=int(image_size))
    dfx_val = pd.read_csv(x_csv[1], index_col=0)
    x_val = __reshape(dfx_val.values, rows=int(image_size), cols=int(image_size))
    dfx_test = pd.read_csv(x_csv[2], index_col=0)
    x_test = __reshape(dfx_test.values, rows=int(image_size), cols=int(image_size))
    dfy_train = pd.read_csv(y_csv[0], index_col=0)
    y_train = dfy_train.values
    dfy_val = pd.read_csv(y_csv[1], index_col=0)
    y_val = dfy_val.values
    dfy_test = pd.read_csv(y_csv[2], index_col=0)
    y_test = dfy_test.values
    return x_train, x_val, x_test, y_train, y_val, y_test


def create_X(filepath):
    stft_data = compute_STFT(filepath, sr=sampling_rate, sample_duration=sampling_period, offset=0)
    display_spectogram(data=stft_data, sampling_rate=sampling_rate, title=filepath)
    data = create_grey_spectogram(data=stft_data,sampling_rate=sampling_rate, image_size=image_size)
    #data = np.ravel(data)
    return data


def create_Y(y_value):
    return genre == y_value


def __reshape(matrix, rows, cols, blocks=-1, depth=1):
    result = matrix.reshape(blocks, rows, cols, depth)
    return result
