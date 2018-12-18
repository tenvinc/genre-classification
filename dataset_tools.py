from __future__ import print_function

import os

import numpy as np

import logs_centre
from audio_processing import create_mel_ndarray

# Default sampling parameters
sampling_period = 30.0
sampling_rate = 44100  # in Hz

# Default spectogram parameters
image_size = 256

# Default folder parameters
RAW_DATA_FOLDER = 'raw_data'
DATASET_FOLDER = 'dataset'

# Dataset class range
GENRE = np.asarray(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
                    'pop', 'reggae', 'rock'])

# Default train:val:test ratio which adds up to 1
SPLIT_RATIO = (0.6, 0.2, 0.2)
MAX_EXAMPLES = 1000

# Storage parameters
x_npy = ('trainX.npy', 'valX.npy', 'testX.npy')
y_npy = ('trainY.npy', 'valY.npy', 'testY.npy')
label_npy = ('labelTrain.npy', 'labelVal.npy', 'labelTest.npy')

logger = logs_centre.get_logger(__name__)


def prepare_dataset(path):
    search_path = os.path.join(path, RAW_DATA_FOLDER)
    logger.info('============ Attempting to start the conversion =======================')
    x_combined, y_combined, labels_combined = create_dataset(search_path)
    logger.info('============ Finished. Attempting to split dataset ====================')
    x_split, y_split, label_split = split_dataset(ratios=SPLIT_RATIO, X=x_combined, Y=y_combined,
                                                  labels=labels_combined,
                                                  total=x_combined.shape[0])
    logger.info('============ Finished. Attempting to save dataset to disk =============')
    save_dataset(x_split, y_split, label_split)


def create_dataset(search_path):
    dataset_x = []
    dataset_y = []
    labels = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(search_path):
        for filename in filenames[:1]:
            y_value = os.path.basename(dirpath)
            logger.info('Processing {} now'.format(filename))
            X = create_X(os.path.join(dirpath, filename), count)
            Y = create_Y(y_value)

            logger.info('Attempting to add generated data and label to list...')
            labels.append(filename)
            dataset_x.append(X)
            dataset_y.append(Y)
            count += 1
            logger.info('Moving on to the next file...')
            print('.', end='')
        print(' {}%'.format((count / MAX_EXAMPLES) * 100))
    logger.info('Finished creation of dataset.')
    return np.array(dataset_x), np.array(dataset_y), np.array(labels)


def split_dataset(ratios, X, Y, labels, total):
    num_example = X.shape[0]
    logger.info('Attempting to split {} sets of data'.format(num_example))
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
    logger.info('Completed the splitting of data')
    return x, y, split_labels


def save_dataset(x_split, y_split, label_split):
    if (len(x_split) != len(y_split)) or (len(x_split) != len(label_split)):
        logger.error('The length of x, y, label are not the same. Aborting save operation...')
    if not os.path.exists(DATASET_FOLDER):
        os.mkdir(DATASET_FOLDER)
    index = 0
    while index < len(x_split):
        np.save(file=os.path.join(DATASET_FOLDER, x_npy[index]), arr=x_split[index])
        np.save(file=os.path.join(DATASET_FOLDER, y_npy[index]), arr=y_split[index])
        np.save(file=os.path.join(DATASET_FOLDER, label_npy[index]), arr=label_split[index])
        logger.info('Saving data to {}, {}, {} in {}'.format(x_npy[index], y_npy[index],
                                                             label_npy[index], DATASET_FOLDER))
        index += 1


# Reads in the datasets from csv files, ignoring the first row and first col of the data
def read_dataset():
    logger.info('Attempting to read dataset from disk...')
    try:
        x_train = np.load(os.path.join(DATASET_FOLDER, x_npy[0]))
        x_val = np.load(os.path.join(DATASET_FOLDER, x_npy[1]))
        x_test = np.load(os.path.join(DATASET_FOLDER, x_npy[2]))
        y_train = np.load(os.path.join(DATASET_FOLDER, y_npy[0]))
        y_val = np.load(os.path.join(DATASET_FOLDER, y_npy[1]))
        y_test = np.load(os.path.join(DATASET_FOLDER, y_npy[2]))
    except FileNotFoundError:
        logger.error('Dataset folder <{}> not found. Aborting operation...'.format(DATASET_FOLDER))
        raise
    logger.info('Finished reading dataset from data.')
    return x_train, x_val, x_test, y_train, y_val, y_test


def create_X(filepath, count):
    feature = create_mel_ndarray(filepath, count)
    return feature


def create_Y(y_value):
    return (GENRE == y_value).astype(int)
