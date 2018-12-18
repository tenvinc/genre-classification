from __future__ import print_function

import os

import librosa as lbr
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import logs_centre

# This module contains functions to process audio data into spectograms and returns a numpy array of values for the
# dataset

# Default image parameters to make figure size easy to set
dpi = 256.0
DEFAULT_FIG_SIZE = (1, 1)

# Audio input parameters
SAMPLING_RATE = 44100  # in Hz
DURATION = 10.0 # in seconds
OFFSET = 5.0 # in seconds

# Mel spectogram characteristic params
MAX_FREQ = 8000
WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MEL = 128

logger = logs_centre.get_logger(__name__)


def create_mel_ndarray(path, index):
    logger.info('Attempting to create mel spectogram ndarray')
    mel_matrix = compute_mel(path)
    fig = draw_melspectogram(mel_matrix, DEFAULT_FIG_SIZE)
    result = convert_mel_to_nparray2d(fig, index)
    logger.info('Finished creation of mel spectogram ndarray')
    return result


# Returns Short-time Fourier Transform of an audio file in terms of mel scale
def compute_mel(path):
    y, sr = lbr.load(path=path, sr=SAMPLING_RATE, mono=True, duration=DURATION, offset=OFFSET)
    mel_matrix = lbr.feature.melspectrogram(y=y, sr=SAMPLING_RATE,
                                            n_fft=WINDOW_SIZE, hop_length=WINDOW_STRIDE,
                                            fmax=MAX_FREQ)
    return mel_matrix


def draw_melspectogram(mel_matrix, fig_shape):
    fig = plt.figure(figsize=fig_shape, dpi=dpi)
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # hide axes
    lbr.display.specshow(lbr.power_to_db(mel_matrix, ref=np.max),
                         y_axis='mel', fmax=MAX_FREQ,
                         x_axis='time', cmap='gray')
    return fig


def convert_mel_to_nparray2d(fig, index):
    rgb_data = __convert_to_numpy(fig, index)
    greyscale_data = __rgb2gray(rgb_data)
    return greyscale_data


def __convert_to_numpy(fig, index):
    fig.savefig(os.path.join('images', 'temp{}.png'.format(index)))
    np_data = np.array(fig.canvas.renderer._renderer)
    plt.close('all')
    return np_data


def __rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def display_spectogram(data, sampling_rate, title, win_length=2048):
    librosa.display.specshow(data, sr=sampling_rate, y_axis='log', x_axis='time', hop_length=win_length/4)
    plt.colorbar(format='%0.2f dB')
    plt.title(title)
    plt.show()
    plt.close()























