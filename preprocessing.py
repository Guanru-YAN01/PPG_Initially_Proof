import numpy as np
from scipy.signal import detrend, butter, filtfilt, hilbert

def detrend_signal(signal: np.ndarray) -> np.ndarray:
    return detrend(signal)

def remove_baseline(signal: np.ndarray, window_size: int) -> np.ndarray:
    window = np.ones(window_size) / window_size
    baseline = np.convolve(signal, window, mode='same')
    return signal - baseline

def butterworth_lowpass(signal: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def normalize_minmax(signal: np.ndarray) -> np.ndarray:
    """
    Min-max normalize the signal to the range [-1, 1].
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return np.zeros_like(signal)
    return 2 * (signal - min_val) / (max_val - min_val) - 1

def normalize_envelope(signal: np.ndarray) -> np.ndarray:
    """
    Normalize by dividing by the analytic signal envelope, then min-max to [-1,1].
    """
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    # avoid division by zero
    envelope[envelope == 0] = 1e-8
    normed = signal / envelope
    return normalize_minmax(normed)