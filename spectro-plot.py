import matplotlib.pyplot as plt
import numpy as np

def spectro_plot(Y, x_array, y_array, title, xlabel, ylabel):
    """

    Parameters
    ----------
    Y : input, type ndarray, frame-by-frame STFT frequency spectrum returned by spectro-mat()
    x_array : type ndarray, x-axis values
    y_array : type ndarray, y-axis values
    title : type str, spectrogram title
    xlabel : type str, x-axis label
    ylabel : type str, y-axis label

    Prints : type plt.pcolormesh, spectrogram frequency visualization
    Returns nothing
    -------

    """
    plt.pcolormesh(x_array, y_array, Y)
    plt.title("title")
    plt.xlabel("xlabel")
    plt.ylabel("ylabel")
    plt.colorbar()