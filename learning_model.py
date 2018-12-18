import os

import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import Sequence
from sklearn.preprocessing import StandardScaler

import logs_centre
from dataset_tools import image_size

# Set this to true to use GPU for tensorflow
IS_GPU_USED = False

# Model parameters (Hyper parameters)
INIT_LEARNING_RATE = 0.001
DROPOUT_CONST = 0.5

logger = logs_centre.get_logger(__name__)


def setup_tensorflow():
    num_cores = 2
    if IS_GPU_USED:
        logger.info('Selecting GPU...')
        num_CPU = 0
        num_GPU = 1
    else:
        logger.info('Selecting CPU...')
        num_CPU = 1
        num_GPU = 0

    # TensorFlow wizardry
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores,
                            allow_soft_placement=True,
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})

    if IS_GPU_USED:
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 1.0

    # Create a session with the above options specified.
    k.set_session(tf.Session(config=config))
    logger.info('Finished configuring TensorFlow settings.')


def create_model(image_size, class_count):
    logger.info('Attempting to create CNN model...')
    model = Sequential()
    model.add(Reshape((image_size, image_size, 1), input_shape=(image_size, image_size)))

    # Hidden layer 1
    model.add(Conv2D(128, kernel_size=3, activation='relu', strides=1, padding='same'))
    model.add(MaxPooling2D(3,4))
    model.add(Dropout(DROPOUT_CONST))

    # Hidden layer 2
    model.add(Conv2D(256, kernel_size=3, activation='relu', strides=1, padding='same'))
    model.add(MaxPooling2D(4, 5))
    model.add(Dropout(DROPOUT_CONST))

    logger.info('Still creating CNN model...')
    # Hidden layer 3
    model.add(Conv2D(512, kernel_size=3, activation='relu', strides=1, padding='same'))
    model.add(MaxPooling2D(5, 6))
    model.add(Dropout(DROPOUT_CONST))

    # Hidden layer 4
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))

    # Output layer
    model.add(Dense(class_count, activation='softmax'))

    # Compile settings
    opt = Adam(lr=INIT_LEARNING_RATE)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    logger.info('Model successfully created...')
    return model


def train_model(model, sequence, data_val, label_val, epochs):
    logger.info('Attempting to train model with data...')
    history = model.fit_generator(generator=sequence, epochs=epochs, validation_data=(data_val, label_val))
    logger.info('Finished training model.')
    return model, history


def preprocess_data(data, scaler=None):
    logger.info('Attempting to preprocess data...')
    new_scaler = scaler
    scaled_data = []
    for d in data:
        d = d.reshape(-1, image_size * image_size)
        if new_scaler is None:
            logger.info('No scaler object detected. Creating one now...')
            new_scaler = StandardScaler().fit(d)
            d = new_scaler.transform(d)
        else:
            logger.info('Scaler object detected. Applying same transformation on data...')
            d = new_scaler.transform(d)
        scaled_data.append(d.reshape(-1, image_size, image_size))
    logger.info('Finished preprocessing of data.')
    return new_scaler, scaled_data


def save_model(model, title):
    folder = 'models'
    if not os.path.exists(folder):
        print('Directory missing. Creating {} '.format(folder))
        os.mkdir(folder)
    print('Attempting to save to {}'.format(os.path.join(folder, title) + '.h5'))
    model.save(os.path.join(folder, title) + '.h5')
    print('Saved to {}'.format(os.path.join(folder, title) + '.h5'))


class SpectogramSequence(Sequence):

    def __init__(self, x_dataset, y_dataset, batch_size):
        logger.info('Setting up spectogram sequence of {} per batch'.format(batch_size))
        self.x, self.y = x_dataset, y_dataset
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
