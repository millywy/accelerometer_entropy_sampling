"""Per-window ACC feature extraction for the PPG HR-estimation pipeline.

Parallel to `hr_estimator.py`. Exposes the motion-only (gravity-removed) RMS of
the 3-axis accelerometer magnitude, computed per 8 s window.

Pre-processing pipeline (per window):
    1. Slice 3 × WIN_SAMPLES segment out of the 3 × N recording.
    2. Subtract each axis's own mean inside the window — removes the ~1 g gravity
       DC component so that a stationary wrist reads ≈ 0, not ≈ 1.
    3. Compute per-sample 3-axis magnitude: sqrt(ax² + ay² + az²).
    4. Take the window's RMS of that magnitude.

This isolates motion energy from the constant gravity pull and is the
per-window ACC feature used as a predictor of HR estimation error in
Step 3 and downstream.

Typical usage::

    from acc_features import per_window_rms_ac, attach_ac_rms
    ac_rms = per_window_rms_ac(rec['acc'])          # 1-D, length = n_windows
    attach_ac_rms(recordings)                        # adds fields in-place
"""

from __future__ import annotations

import numpy as np

from data_loader import FS, WIN_SAMPLES, SHIFT


def demean_window(acc_seg):
    """Subtract each axis's own mean inside a 3 × WIN segment.

    Returns a 3 × WIN array with per-axis mean == 0.
    """
    return acc_seg - acc_seg.mean(axis=1, keepdims=True)


def rms_magnitude_ac(acc_seg):
    """AC-only RMS of the 3-axis magnitude for one 3 × WIN segment.

    Demean each axis → per-sample magnitude → window RMS.
    """
    seg  = demean_window(acc_seg)
    mag  = np.sqrt(np.sum(seg**2, axis=0))
    return float(np.sqrt(np.mean(mag**2)))


def per_window_rms_ac(acc, win=WIN_SAMPLES, shift=SHIFT):
    """Sliding-window AC-only RMS across a full 3 × N recording.

    acc:   3 × N array of raw ACC samples (gravity DC still present).
    win:   samples per window (default 1000 = 8 s at 125 Hz).
    shift: samples between windows (default 250 = 2 s).

    Returns a 1-D array of length `n_windows` of motion-only RMS values.
    """
    n_windows = (acc.shape[1] - win) // shift + 1
    rms = np.empty(n_windows)
    for i in range(n_windows):
        seg = acc[:, i*shift : i*shift + win]
        rms[i] = rms_magnitude_ac(seg)
    return rms


def attach_ac_rms(recordings, win=WIN_SAMPLES, shift=SHIFT):
    """Mutate each recording dict in-place to carry AC-only RMS features.

    Adds the following keys:
      `acc_rms_ac_windows` — 1-D array, AC-only RMS per window
      `mean_acc_rms_ac`    — scalar, mean across windows

    Returns the same list for chaining.
    """
    for rec in recordings:
        ac = per_window_rms_ac(rec['acc'], win=win, shift=shift)
        rec['acc_rms_ac_windows'] = ac
        rec['mean_acc_rms_ac']    = float(ac.mean())
    return recordings
