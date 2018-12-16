from __future__ import print_function
from audio_processing import create_mel_ndarray
import numpy as np
import os

# Default sampling parameters
sampling_period = 30.0
sampling_rate = 44100      # in Hz

# Default spectogram parameters
image_size = 256.0

# Default folder parameters
FOLDER_NAME = 'dataset'

# Dataset class range
GENRE = np.asarray(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
                    'pop', 'reggae', 'rock'])

# Default train:val:test ratio which adds up to 1
SPLIT_RATIO = (0.6, 0.2, 0.2)
MAX_EXAMPLES = 1000

# Default data storage names
x_npy = ('trainX.npy', 'valX.npy', 'testX.npy')
y_npy = ('trainY.npy', 'valY.npy', 'testY.npy')


def prepare_dataset(path):
    search_path = os.path.join(path, FOLDER_NAME)
    print('============ Attempting to start the conversion...')
    x_combined, y_combined = create_dataset(search_path)
    print('============ Finished. Attempting to split dataset...')
    x_split, y_split = split_dataset(ratios=SPLIT_RATIO, X=x_combined, Y=y_combined,
                                     total=x_combined.shape[0])
    print('============ Finished. Attempting to save dataset to disk...')
    save_dataset(x_split, y_split)


def create_dataset(search_path):
    dataset_x = []
    dataset_y = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(search_path):
        for filename in filenames[:1]:
            X = create_X(os.path.join(dirpath, filename), count)
            y_value = os.path.basename(dirpath)
            Y = create_Y(y_value)
            dataset_x.append(X)
            dataset_y.append(Y)
            count += 1
            print('.', end='')
        print(' {}%'.format((count / MAX_EXAMPLES) * 100))
    return np.array(dataset_x), np.array(dataset_y)


def split_dataset(ratios, X, Y, total):
    num_example = X.shape[0]
    random_indices = np.random.permutation(num_example)
    x = []
    y = []
    end = 0
    for ratio in ratios:
        start = end
        end = start + int(ratio * total)
        x.append(X[random_indices[start:end]])
        y.append(Y[random_indices[start:end]])
    return x, y


def save_dataset(x_split, y_split):
    i = 0
    while i < len(x_split):
        np.save(file=x_npy[i], arr=x_split[i])
        print('x = {}'.format(x_split[i].shape))
        print('saving to {}'.format(x_npy[i]))
        i += 1
    j = 0
    while j < len(y_split):
        np.save(file=y_npy[j], arr=y_split[j])
        print('y = {}'.format(y_split[j].shape))
        print('saving to {}'.format(y_npy[j]))
        j += 1


# Reads in the datasets from csv files, ignoring the first row and first col of the data
def read_dataset():
    x_train = np.load(x_npy[0])
    x_val = np.load(x_npy[1])
    x_test = np.load(x_npy[2])
    y_train = np.load(y_npy[0])
    y_val = np.load(y_npy[1])
    y_test = np.load(y_npy[2])
    return x_train, x_val, x_test, y_train, y_val, y_test


def create_X(filepath, count):
    feature = create_mel_ndarray(filepath, count)
    return feature


def create_Y(y_value):
    return GENRE == y_value


def __reshape(matrix, rows, cols, blocks=-1, depth=1):
    result = matrix.reshape(blocks, rows, cols, depth)
    return result
