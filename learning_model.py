from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Dropout
from keras.utils import Sequence
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as k
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataset_tools import image_size
import os

# Set this to true to use GPU for tensorflow
IS_GPU_USED = False


def setup_tensorflow():
    num_cores = 2
    if IS_GPU_USED:
        num_CPU = 0
        num_GPU = 1
    else:
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


def create_model(image_size, class_count):
    print('Attempting to create CNN model...', flush=True)
    model = Sequential()
    model.add(Reshape((256, 256, 1), input_shape=(256, 256)))

    # Hidden layer 1
    model.add(Conv2D(64, kernel_size=3, activation='relu', strides=2))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))

    # Hidden layer 2
    model.add(Conv2D(128, kernel_size=3, activation='relu', strides=2))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))

    # Hidden layer 3
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))

    # Output layer
    model.add(Dense(class_count, activation='softmax'))

    # Compile settings
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    print('Model successfully created...', flush=True)
    return model


def train_model(model, sequence, data_val, label_val, iter):
    print('Training model with data...', flush=True)
    history = model.fit_generator(generator=sequence,epochs=iter, validation_data=(data_val, label_val))
    return model, history


def preprocess_data(data, scaler=None):
    new_scaler = scaler
    scaled_data = []
    for d in data:
        d = d.reshape(-1, int(image_size * image_size))
        if new_scaler is None:
            new_scaler = StandardScaler().fit(d)
            d = new_scaler.transform(d)
        else:
            d = new_scaler.transform(d)
        scaled_data.append(d.reshape(-1, int(image_size), int(image_size)))
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
        self.x, self.y = x_dataset, y_dataset
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
