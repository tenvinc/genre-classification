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
label_npy = ('labelTrain.npy', 'labelVal.npy', 'labelTest.npy')


def prepare_dataset(path):
    search_path = os.path.join(path, FOLDER_NAME)
    print('============ Attempting to start the conversion...')
    x_combined, y_combined, labels_combined = create_dataset(search_path)
    print('============ Finished. Attempting to split dataset...')
    x_split, y_split, label_split = split_dataset(ratios=SPLIT_RATIO, X=x_combined, Y=y_combined,
                                                  labels=labels_combined,
                                                  total=x_combined.shape[0])
    print('============ Finished. Attempting to save dataset to disk...')
    save_dataset(x_split, y_split, label_split)


def create_dataset(search_path):
    dataset_x = []
    dataset_y = []
    labels = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(search_path):
        for filename in filenames:
            y_value = os.path.basename(dirpath)
            X = create_X(os.path.join(dirpath, filename), count)
            Y = create_Y(y_value)
            labels.append(filename)
            dataset_x.append(X)
            dataset_y.append(Y)
            count += 1
            print('.', end='')
        print(' {}%'.format((count / MAX_EXAMPLES) * 100))
    return np.array(dataset_x), np.array(dataset_y), np.array(labels)


def split_dataset(ratios, X, Y, labels, total):
    num_example = X.shape[0]
    random_indices = np.random.permutation(num_example)
    x = []
    y = []
    split_labels = []
    end = 0
    for ratio in ratios:
        start = end
        end = start + int(ratio * total)
        x.append(X[random_indices[start:end]])
        y.append(Y[random_indices[start:end]])
        split_labels.append(labels[random_indices[start:end]])
    return x, y, split_labels


def save_dataset(x_split, y_split, label_split):
    i = 0
    while i < len(x_split):
        np.save(file=x_npy[i], arr=x_split[i])
        print('saving to {}'.format(x_npy[i]))
        i += 1
    j = 0
    while j < len(y_split):
        np.save(file=y_npy[j], arr=y_split[j])
        print('saving to {}'.format(y_npy[j]))
        j += 1
    k = 0
    while k < len(label_split):
        np.save(file=label_npy[k], arr=label_split[k])
        print('saving to {}'.format(label_npy[k]))
        k += 1


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
    return (GENRE == y_value).astype(int)


def __reshape(matrix, rows, cols, blocks=-1, depth=1):
    result = matrix.reshape(blocks, rows, cols, depth)
    return result
