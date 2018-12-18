import matplotlib.pyplot as plt

import logs_centre

logger = logs_centre.get_logger(__name__)

def plot_acc_wrt_epochs(history):
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    save_plot(fig)
    plt.clf()


def plot_loss_wrt_epochs(history):
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    save_plot(fig)
    plt.clf()


def save_plot(fig):
    while True:
        response = input('Do you want to save the current plot? (y/n)')
        if response == 'n':
            break
        elif response == 'y':
            title = input('Please enter the filename:')
            fig.savefig(title + '.png')
            logger.info('Saved current plot to {}'.format(title + '.png'))
            break
