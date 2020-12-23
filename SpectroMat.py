import numpy as np
from scipy import signal

# Helper function that transforms input into buff_sz-by-1+(N-buff_sz)//hop_length matrix (** EXTRA CREDIT 2 **)

def hoppy(x, frame_length=1024, hop_length=512):
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


# Main function for assignment

def spectrum(x, fs, buff_sz, win, pad_len=0):
    # Use helper function to create non-overlapping matrix if win == "rect"
    if win.lower() == "rect":
        xframes = hoppy(x, frame_length=buff_sz, hop_length=buff_sz)

    # Otherwise, use helper function to allow for matrix with hop_length parameter control (hop_length=512 by default)
    else:
        xframes = hoppy(x, frame_length=buff_sz, hop_length=buff_sz // 2)

    # Set up zero-valued output matrix, Y, with space provided for the RFFT and pad_len
    Y = np.zeros([(len(xframes[:, 0]) + pad_len) // 2 + 1, len(xframes[0, :])])

    # Add zero-padding beyond buff_sz (** EXTRA CREDIT 1 **)
    xframes = np.pad(xframes, ((0, pad_len), (0, 0)))

    # Create the desired window, ignoring any zero-padding
    w = signal.get_window(win.lower(), buff_sz)

    # For each frame:
    for i in range(len(xframes[0, :])):
        # Window each frame individally, ignoring any zero-padding
        xframes[:buff_sz, i] *= w

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