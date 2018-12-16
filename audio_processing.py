from __future__ import print_function
import librosa.display
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# This module contains functions to process audio data into spectograms and returns a numpy array of values for the
# dataset

# Default image parameters
dpi = 100.0


# Returns Short-time Fourier Transform of the file given a path, sampling rate, offset and duration of sample
def compute_STFT(path, sr, sample_duration, n_fft=2048, offset=None):
    total_duration = librosa.get_duration(filename=path)
    if offset is None:
        offset = total_duration / 2
    if (offset + sample_duration) > total_duration:
        print("Error. Duration of audio is too little")
        exit(1)
    y, sr = librosa.load(path=path, sr=sr)
    data = librosa.amplitude_to_db(np.abs(librosa.stft(y=y)), ref=np.max)
    return data


# Creates a numpy array of pixel values from a greyscaled spectogram
def create_grey_spectogram(data, sampling_rate,  image_size, win_length=2048):
    fig = __plot_grey_spectogram(data, sampling_rate, image_size, win_length)
    rgb_data = __convert_to_numpy(fig)
    greyscale_data = __rgb2gray(rgb_data)
    return greyscale_data


def __plot_grey_spectogram(data, sampling_rate, image_size, win_length):
    fig = plt.figure(figsize=(image_size/dpi, image_size/dpi), dpi=dpi)
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # hide axes
    librosa.display.specshow(data, sr=sampling_rate, x_axis='off', y_axis='log',
                             hop_length=win_length/4)
    # Currently requires saving before figure is saved. TODO: change this in the future
    plt.savefig('demo.png', bbox_inches=None, pad_inches=0)
    plt.close()
    return fig


def __convert_to_numpy(fig):
    np_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    np_data = np_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
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
