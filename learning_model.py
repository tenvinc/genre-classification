from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import Sequence
import tensorflow as tf
from keras import backend as k
import numpy as np


def setup_tensorflow():
    num_cores = 2
    num_CPU = 1
    num_GPU = 0

    # TensorFlow wizardry
    config = config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True, \
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})

    # Don't pre-allocate memory; allocate as-needed
    # config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 1.0

    # Create a session with the above options specified.
    k.set_session(tf.Session(config=config))


def create_model(image_size, class_count):
    print('Attempting to create CNN model...', flush=True)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=2, activation='relu', input_shape=(image_size, image_size, 1), strides=1,
                     padding='same'))
    model.add(Conv2D(128, kernel_size=2, activation='relu', strides=1, padding='same'))
    model.add(Conv2D(256, kernel_size=2, activation='relu', strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dense(class_count, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print('Model successfully created...', flush=True)
    return model


def train_model(model, sequence, data_val, label_val, iter):
    print('Training model with data...', flush=True)
    history = model.fit_generator(generator=sequence, epochs=iter, validation_data=(data_val, label_val))
    return model, history


class SpectogramSequence(Sequence):

    def __init__(self, x_dataset, y_dataset, batch_size):
        self.x, self.y = x_dataset, y_dataset
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
