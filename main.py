# Beat tracking example
from __future__ import print_function

import os

import logs_centre
from dataset_tools import prepare_dataset, read_dataset, image_size, GENRE
from evaluation_tools import plot_acc_wrt_epochs, plot_loss_wrt_epochs
from learning_model import SpectogramSequence as ss
from learning_model import setup_tensorflow, create_model, preprocess_data, train_model, save_model

# Set this param to set the mode
is_training = True
is_generating_data = False

# Machine learning params
BATCH_SIZE = 20  # batch size to be fed into each iteration
EPOCHS_COUNT = 20  # number of epochs to be run

logger = logs_centre.get_logger(__name__)
search_path = os.getcwd()

# Generates data sets and saves them in a folder
if is_generating_data:
    logger.info('Generating Data mode detected. Generating data now...')
    prepare_dataset(search_path)
    logger.info('Finished generation and storage of dataset.')
if is_training:
    logger.info('Training mode detected.')
    x_train, x_val, x_test, y_train, y_val, y_test = read_dataset()
    setup_tensorflow()

    model = create_model(image_size, GENRE.shape[0])
    logger.info(model.summary())
    scaler, (x_train, x_val, x_test) = preprocess_data((x_train, x_val, x_test))
    sequence = ss(x_train, y_train, batch_size=BATCH_SIZE)
    model, history = train_model(model, sequence, data_val=x_val, label_val=y_val, epochs=EPOCHS_COUNT)

    plot_acc_wrt_epochs(history)
    plot_loss_wrt_epochs(history)

    while True:
        prompt = input('Do you want to save this model? (y/n)')
        if prompt == 'n':
            break
        elif prompt == 'y':
            title = input('Please enter a title for the model to be saved under: ')
            save_model(model, title)
            break
        else:
            print('Invalid command. Please try again...')


