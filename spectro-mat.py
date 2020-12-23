import numpy as np
from scipy import signal

# Helper function that transforms input into frame_length-by-1+(N-frame_length)//hop_length matrix (** EXTRA CREDIT 2 **)

def hoppy(x, frame_length=1024, hop_length=512):
    """
    Helper function for spectro_mat().
    Slices a time series into overlapping frames.

    Parameters
    ----------
    x : np.ndarray [shape=(n,)]
        Time series to frame. Must be one-dimensional and contiguous
        in memory.

    frame_length : int > 0 [scalar]
        Length of the frame in samples

    hop_length : int > 0 [scalar]
        Number of samples to hop between frames

    Returns
    -------
    y_frames : np.ndarray [shape=(frame_length, N_FRAMES)]
        An array of frames sampled from `x`:
        `y_frames[i, j] == y[j * hop_length + i]`

    Raises
    ------
    ParameterError
        If `y` is not contiguous in memory, not an `np.ndarray`, or
        not one-dimensional.  See `np.ascontiguous()` for details.

        If `hop_length < 1`, frames cannot advance.

        If `len(y) < frame_length`
    """

    if not isinstance(x, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                             'given type(y)={}'.format(type(x)))

    if x.ndim != 1:
        raise ParameterError('Input must be one-dimensional, '
                             'given y.ndim={}'.format(x.ndim))

    if len(x) < frame_length:
        raise ParameterError('Buffer is too short (n={:d})'
                             ' for frame_length={:d}'.format(len(x), frame_length))

    if hop_length < 1:
        raise ParameterError('Invalid hop_length: {:d}'.format(hop_length))

    if not x.flags['C_CONTIGUOUS']:
        raise ParameterError('Input buffer must be contiguous.')

    N = len(x)
    num_frames = 1 + (N - frame_length) // hop_length
    y = np.zeros([frame_length, num_frames])
    N = y.size

    for n in range(int(N // frame_length)):
        y[:, n] += x[n * hop_length:n * hop_length + frame_length]
    return y

# Main function

def spectro_mat(x, fs, frame_length, win, pad_len=0):
    """
    Converts time-domain array into frequency-domain matrix with a frame-by-frame STFT implementation.

    Parameters
    ----------
    x : np.ndarray [shape=(n,)]
        Time series to frame. Must be one-dimensional and contiguous
        in memory.

    fs : int
        Sample rate.

    frame_length : int
        Buffer size.

    win : str
        The type of window to be integrated into FFT computation

    pad_len : int
        Number of zeros to be added to each frame before the FFT is computed

    Returns
    -------
    Y : np.ndarray [shape=(frame_length+pad_len//2 + 1, N_FRAMES)]
        A windowed, frame-by-frame frequency-domain representation of x, calculated via FFT
    """

    # Use helper function to create non-overlapping matrix if win == "rect"
    if win.lower() == "rect":
        xframes = hoppy(x, frame_length=frame_length, hop_length=frame_length)

    # Otherwise, use helper function to allow for matrix with hop_length parameter control (hop_length=512 by default)
    else:
        xframes = hoppy(x, frame_length=frame_length, hop_length=frame_length // 2)

    # Set up zero-valued output matrix, Y, with space provided for the RFFT and pad_len
    Y = np.zeros([(len(xframes[:, 0]) + pad_len) // 2 + 1, len(xframes[0, :])])

    # Add zero-padding beyond frame_length (** EXTRA CREDIT 1 **)
    xframes = np.pad(xframes, ((0, pad_len), (0, 0)))

    # Create the desired window, ignoring any zero-padding
    w = signal.get_window(win.lower(), frame_length)

    # For each frame:
    for i in range(len(xframes[0, :])):
        # Window each frame individually, ignoring any zero-padding
        xframes[:frame_length, i] *= w

        # Compute real-valued FFT, including any zero-padding
        Y[:, i] += np.abs(np.fft.rfft(xframes[:, i]))

        # Normalize in frequency domain
        Y[:, i] /= Y[:, i].size

        # Convert to dB
        Y[:, i] = 20 * np.log10(Y[:, i])

    # Create x- and y-axis arrays
    t_array = np.linspace(0, Y.size / fs, len(Y[0, :]))
    f_array = np.linspace(0, fs // 2, len(Y[:, 0]))

    # Return output
    return Y, f_array, t_array