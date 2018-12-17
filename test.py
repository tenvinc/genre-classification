# Beat tracking example
from __future__ import print_function

from audio_processing import compute_mel

from dataset_tools import prepare_dataset
from dataset_tools import read_dataset

from learning_model import setup_tensorflow
from learning_model import create_model
from learning_model import train_model
from learning_model import SpectogramSequence as ss
from learning_model import preprocess_data
from learning_model import save_model

import numpy as np

import librosa as lbr
import librosa.display
import os
import debug
import matplotlib.pyplot as plt

# Set this param to set the mode
is_training = True
is_generating_data = False


# Mel spectogram characteristic params
MAX_FREQ = 8000
WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MEL = 128


search_path = os.getcwd()

if is_generating_data:
    prepare_dataset(search_path)
if is_training:
    x_train, x_val, x_test, y_train, y_val, y_test = read_dataset()
    label_train, label_val, label_test = debug.read_dataset()
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)

    setup_tensorflow()
    model = create_model(256, 10)
    print(model.summary())
    title = input('Please enter your desired title for the model:')
    scaler, (x_train, x_val, x_test) = preprocess_data((x_train, x_val, x_test))
    sequence = ss(x_train, y_train, batch_size=20)
    model, history = train_model(model, sequence, data_val=x_val,
                                 label_val=y_val, iter=10)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    print('Predicted')
    print(model.predict(x_test[0].reshape(1, 256, 256)))
    print('Actual:')
    print(y_test[0])

    save_model(model, title)




