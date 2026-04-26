"""Stateless per-window HR estimator for PPG signals.

Pipeline (matches the Step 2 spec):

    1. Bandpass-filter each PPG channel (Butterworth, 0.5-4 Hz = 30-240 BPM).
    2. Average the two channels into a single 1-D signal.
    3. Slice into 8 s windows with 2 s shift.
    4. For each window, compute zero-padded FFT and pick the dominant peak
       in [0.5, 4] Hz; convert peak frequency to BPM (×60).

The estimator is STATELESS — each window's BPM is produced without any
information from previous windows. Filtering is done end-to-end on the
whole recording once (so per-window filter transients are avoided), but
that is a static per-recording operation, not history across windows.

Typical usage::

    from hr_estimator import estimate_hr_trace, evaluate_recording
    est_bpm = estimate_hr_trace(rec['ppg'])
    result  = evaluate_recording(rec)   # {est_bpm, true_bpm, abs_error, mae}
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from data_loader import FS, WIN_SAMPLES, SHIFT


# ── Default hyper-parameters ─────────────────────────────────────────────────
HR_LO_HZ   = 0.5          # 30 BPM
HR_HI_HZ   = 4.0          # 240 BPM
FILT_ORDER = 4
N_FFT      = 8192         # zero-padded FFT size — ~0.015 Hz ≈ 0.9 BPM resolution


# ── Bandpass filter ──────────────────────────────────────────────────────────
def design_bandpass(fs=FS, lo=HR_LO_HZ, hi=HR_HI_HZ, order=FILT_ORDER):
    """Butterworth bandpass filter coefficients. Returns (b, a)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return b, a


def bandpass_filter(x, fs=FS, lo=HR_LO_HZ, hi=HR_HI_HZ, order=FILT_ORDER):
    """Zero-phase Butterworth bandpass along the last axis.

    x may be 1-D or 2-D (each row filtered independently).
    """
    b, a = design_bandpass(fs=fs, lo=lo, hi=hi, order=order)
    return filtfilt(b, a, x, axis=-1)


# ── Channel averaging ────────────────────────────────────────────────────────
def average_channels(ppg_filt):
    """Mean of the two PPG channels. Input: 2×N. Output: 1-D of length N."""
    assert ppg_filt.ndim == 2 and ppg_filt.shape[0] == 2, (
        f'Expected 2×N PPG input, got shape {ppg_filt.shape}')
    return ppg_filt.mean(axis=0)


# ── Single-window FFT peak-pick ──────────────────────────────────────────────
def estimate_hr_window(x_window, fs=FS, lo=HR_LO_HZ, hi=HR_HI_HZ, n_fft=N_FFT):
    """Estimate HR (BPM) for one bandpass-filtered 1-D window.

    Picks argmax of |FFT|² inside [lo, hi] Hz and converts to BPM (×60).
    Returns a float.
    """
    spec  = np.fft.rfft(x_window, n=n_fft) #real-valued FFT, length n_fft//2 + 1 = 4097 bins
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs) #corresponding frequencies in Hz
    mask  = (freqs >= lo) & (freqs <= hi)
    power = np.abs(spec[mask]) ** 2 #power spectrum in the [lo, hi] range
    peak_freq = freqs[mask][np.argmax(power)] #frequency of the dominant peak in Hz
    return peak_freq * 60.0

# helper for debugging: plot the spectrum of one window
def window_spectrum(x_window, fs=FS, n_fft=N_FFT):
    """Return (freqs, power) for a single window's zero-padded FFT.

    Convenience for plotting / debugging.
    """
    spec  = np.fft.rfft(x_window, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    power = np.abs(spec) ** 2
    return freqs, power


# ── Full-recording pipeline ──────────────────────────────────────────────────
def estimate_hr_trace(ppg, fs=FS, win=WIN_SAMPLES, shift=SHIFT,
                      lo=HR_LO_HZ, hi=HR_HI_HZ, order=FILT_ORDER, n_fft=N_FFT):
    """Full pipeline: bandpass → average → per-window FFT peak-pick.

    ppg: 2×N raw PPG (both channels).

    Returns a 1-D array of BPM values, one per window, aligned with the
    BPM0 ground truth produced by the IEEE SPC 2015 protocol.
    """
    ppg_filt = bandpass_filter(ppg, fs=fs, lo=lo, hi=hi, order=order)
    x        = average_channels(ppg_filt)
    n_windows = (x.size - win) // shift + 1
    return np.array([
        estimate_hr_window(x[i*shift : i*shift + win],
                           fs=fs, lo=lo, hi=hi, n_fft=n_fft)
        for i in range(n_windows)
    ])


# ── Evaluation helpers ───────────────────────────────────────────────────────
def evaluate_recording(rec, high_error_threshold=5.0, **kwargs):
    """Run the HR estimator on one recording and compare to its BPM0 ground truth.

    high_error_threshold: BPM. A window is counted as "high error" if its
    absolute error strictly exceeds this value. Default 5 BPM.

    Returns a dict with:
      est_bpm                — 1-D array of estimated BPM per window
      true_bpm               — 1-D array of ground-truth BPM per window
      abs_error              — |est - true| per window
      mae                    — scalar mean absolute error (BPM)
      median_error           — scalar median absolute error (BPM, robust to heavy tails)
      n_high_error           — count of windows with |error| > high_error_threshold
      high_error_threshold   — the threshold used (echoed back for plot labels)
    """
    est  = estimate_hr_trace(rec['ppg'], **kwargs)
    true = rec['bpm0']
    n    = min(len(est), len(true))
    est, true = est[:n], true[:n]
    err  = np.abs(est - true)
    return dict(
        est_bpm=est,
        true_bpm=true,
        abs_error=err,
        mae=float(err.mean()),
        median_error=float(np.median(err)),
        n_high_error=int(np.sum(err > high_error_threshold)),
        high_error_threshold=float(high_error_threshold),
    )


def evaluate_all(recordings, high_error_threshold=5.0, **kwargs):
    """Run the HR estimator on every recording. Returns a list of per-rec dicts.

    Each dict carries rec_id, group, est_bpm, true_bpm, abs_error, mae,
    median_error, n_high_error, high_error_threshold (see `evaluate_recording`).

    Pass `high_error_threshold=...` to override the default 5 BPM threshold.
    """
    results = []
    for rec in recordings:
        r = evaluate_recording(rec, high_error_threshold=high_error_threshold, **kwargs)
        r['rec_id'] = rec['rec_id']
        r['group']  = rec['group']
        results.append(r)
    return results
