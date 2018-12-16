# Beat tracking example
from __future__ import print_function
import os
from dataset_tools import prepare_dataset
from dataset_tools import read_dataset
from dataset_tools import split_dataset
from learning_model import create_model
from learning_model import train_model
from learning_model import setup_tensorflow
from learning_model import SpectogramSequence
from dataset_tools import create_X
from dataset_tools import create_Y
import matplotlib.pyplot as plt
import numpy as np

prepare_dataset(os.getcwd())
'''count = 0
for dirpath, dirnames, filenames in os.walk(os.getcwd()):
    for filename in filenames[:10]:
        count += 1
        X = create_X(os.path.join(dirpath, filename))
        y_value = os.path.basename(dirpath)
        Y = create_Y(y_value)
        print(Y)
        plt.imshow(X)
        plt.show()
    if count >= 10:
        break'''

x_train, x_val, x_test, y_train, y_val, y_test = read_dataset()
for i in range(10):
    print(y_train[i])
    plt.imshow(x_train[i].reshape(200, 200), cmap='gray')
    plt.show()

'''setup_tensorflow()
model = create_model(200, 10)
sequence = SpectogramSequence(x_train, y_train, batch_size=10)
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
plt.legend(['Train', 'Test'], loc='upper left')'''
# plt.show()
