"""Shared data loader for the IEEE SPC 2015 wrist-PPG analysis pipeline.

Imported by step1_data_loading.ipynb and all subsequent step notebooks so that
the 22-recording loader, per-window RMS ACC magnitude, and group colour/label
maps live in exactly one place.

Typical usage from a step notebook::

    from data_loader import load_all_recordings, GROUP_COLORS, GROUP_LABELS, FS
    recordings = load_all_recordings()     # ~2-3 seconds, returns list of 22 dicts

Each recording dict contains::

    rec_id, group, sig_path, bpm_path,
    ecg (None for test files), ppg (2 x N), acc (3 x N), bpm0 (1-D),
    acc_rms_windows, mean_acc_rms, n_windows, n_gt

Three groups:
    T1 -- treadmill        (rec 01-12, training files DATA_*)
    T2 -- mixed arm        (rec 13/14/19/22, test files TEST_S*_T01)
    T3 -- boxing           (rec 15/16/17/18/20/21, test files TEST_S*_T02)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.io as sio


# ── Constants ─────────────────────────────────────────────────────────────────
FS          = 125      # Hz, sampling rate of all signals
WIN_SAMPLES = 1000     # 8 s × 125 Hz
SHIFT       = 250      # 2 s × 125 Hz between successive windows

# Paths are resolved relative to this module so the loader works regardless
# of the notebook's current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent
_DATASET_BASE = _PROJECT_ROOT / 'hpi-dhc TROIKA main datasets-IEEE_SPC_2015'

TRAIN_DIR   = _DATASET_BASE / 'Training_data'
TEST_DIR    = _DATASET_BASE / 'TestData'
TRUEBPM_DIR = _DATASET_BASE / 'TrueBPM'


# ── Group colour and label maps (shared across every plot in the project) ─────
GROUP_COLORS = {
    'T1': 'steelblue',    # treadmill
    'T2': 'goldenrod',    # mixed arm exercise
    'T3': 'crimson',      # boxing
}

GROUP_LABELS = {
    'T1': 'T1 — treadmill',
    'T2': 'T2 — mixed arm exercise',
    'T3': 'T3 — boxing',
}


# ── Manifest ──────────────────────────────────────────────────────────────────
def build_manifest():
    """Enumerate all 22 recordings and assign (rec_id, group) labels.

    Groups:
      T1 — treadmill         (12 training files `DATA_*`)
      T2 — mixed arm         (4  test files `TEST_S*_T01` — shake/stretch/push-up/run/jump)
      T3 — boxing            (6  test files `TEST_S*_T02`)

    Returns a list of dicts with keys {rec_id, group, sig_path, bpm_path}.
    """
    records = []

    # T1: 12 training files with BPMtrace companions
    for sig_path in sorted(TRAIN_DIR.glob('DATA_*.mat')):
        if 'BPMtrace' in sig_path.name:
            continue
        stem     = sig_path.stem                        # e.g. DATA_01_TYPE01
        bpm_path = TRAIN_DIR / f'{stem}_BPMtrace.mat'
        assert bpm_path.exists(), f'Missing BPM file: {bpm_path}'
        rec_id = int(re.search(r'DATA_(\d+)', stem).group(1))
        records.append(dict(rec_id=rec_id, group='T1',
                            sig_path=sig_path, bpm_path=bpm_path))

    # T2/T3: 10 test files — T01 suffix → T2 (mixed arm), T02 suffix → T3 (boxing)
    next_id = 13 #fragile but good enough for this small dataset; just want T1=01-12, T2/T3=13-22
    for sig_path in sorted(TEST_DIR.glob('TEST_S*.mat')):
        m = re.search(r'TEST_(S\d+)_(T\d+)', sig_path.name)
        subject_tag, task_tag = m.group(1), m.group(2)
        bpm_path = TRUEBPM_DIR / f'True_{subject_tag}_{task_tag}.mat'
        assert bpm_path.exists(), f'Missing BPM file: {bpm_path}'
        group = 'T2' if task_tag == 'T01' else 'T3'
        records.append(dict(rec_id=next_id, group=group,
                            sig_path=sig_path, bpm_path=bpm_path))
        next_id += 1

    return records


# ── Single-recording loader ───────────────────────────────────────────────────
def load_recording(rec):
    """Load one recording's signals and normalize the row layout.

    Training `sig` is 6×N: row 0 = ECG, rows 1-2 = PPG, rows 3-5 = ACC.
    Test `sig`    is 5×N: rows 0-1 = PPG, rows 2-4 = ACC (ECG withheld by competition).

    Row indexing is driven by `sig.shape[0]` so it works regardless of how the
    file is labelled; we then assert shape-vs-group consistency to catch any
    manifest/layout drift.

    Returns a dict with keys {ecg, ppg, acc, bpm0}; `ecg` is None for test files.
    """
    sig  = sio.loadmat(rec['sig_path'])['sig'].astype(float)
    bpm0 = sio.loadmat(rec['bpm_path'])['BPM0'].ravel().astype(float) #flatten to 1-D
    n_rows = sig.shape[0]

    if rec['group'] == 'T1':
        assert n_rows == 6, (
            f"Expected 6 rows for T1 training file {rec['sig_path'].name}, got {n_rows}")
    else:
        assert n_rows == 5, (
            f"Expected 5 rows for {rec['group']} test file {rec['sig_path'].name}, got {n_rows}")

    if n_rows == 6:
        ecg = sig[0, :]
        ppg = sig[1:3, :]
        acc = sig[3:6, :]
    else:
        ecg = None
        ppg = sig[0:2, :]
        acc = sig[2:5, :]
    return dict(ecg=ecg, ppg=ppg, acc=acc, bpm0=bpm0)


# ── Per-window RMS ACC magnitude (raw, gravity-DC included) ───────────────────
def rms_acc_windows(acc):
    """Per-window RMS of the 3-axis ACC magnitude.

    acc: 3×N array. Returns a 1-D array of length n_windows. Gravity DC is
    left in the signal here — remove it in a later pre-processing step if
    the analysis needs motion-only energy.
    """
    mag = np.sqrt(np.sum(acc**2, axis=0))
    n_windows = (acc.shape[1] - WIN_SAMPLES) // SHIFT + 1
    rms = np.array([
        np.sqrt(np.mean(mag[i*SHIFT : i*SHIFT + WIN_SAMPLES]**2))
        for i in range(n_windows)
    ])
    return rms


# ── Convenience: load all 22 recordings in one call ───────────────────────────
def load_all_recordings(verbose=False):
    """Load all 22 recordings into a list of dicts ready for downstream analysis.

    Each returned dict contains {rec_id, group, sig_path, bpm_path, ecg, ppg,
    acc, bpm0, acc_rms_windows, mean_acc_rms, n_windows, n_gt}.

    Set verbose=True to print a one-line summary per recording as it loads
    (used by step1_data_loading.ipynb for the sanity-check tour).
    """
    manifest   = build_manifest()
    recordings = []
    for rec in manifest:
        data = load_recording(rec)
        rms  = rms_acc_windows(data['acc'])
        n_gt = len(data['bpm0'])
        recordings.append(dict(
            **rec,
            ecg=data['ecg'],
            ppg=data['ppg'],
            acc=data['acc'],
            bpm0=data['bpm0'],
            acc_rms_windows=rms,
            mean_acc_rms=rms.mean(),
            n_windows=len(rms),
            n_gt=n_gt,
        ))
        if verbose:
            dur_s   = data['acc'].shape[1] / FS
            has_ecg = 'yes' if data['ecg'] is not None else 'no '
            print(f"[{rec['rec_id']:02d}] {rec['group']}  "
                  f"{dur_s:6.1f}s  "
                  f"{len(rms):4d} windows  "
                  f"{n_gt:4d} GT  "
                  f"mean ACC RMS={rms.mean():.3f}  "
                  f"(ECG={has_ecg})")
    return recordings
