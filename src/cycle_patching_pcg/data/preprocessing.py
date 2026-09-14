"""Signal preprocessing and multi-cycle tensor construction.

Ports the module-level functions `butter_bandpass`/`butter_bandpass_filter`
and `retun_multiple_cycles`/`prepare_data` from the original
`Load_multiple_cycles.py` into two small, stateful classes. The numeric
behavior is unchanged -- these are the exact same filter design and
cycle-stacking logic used to produce the paper's results.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


class BandpassFilter:
    """3rd/4th-order Butterworth bandpass filter applied to a raw PCG cycle
    (or batch of cycles). Defaults (25-400 Hz at an assumed 1000 Hz sampling
    rate, order 4) match the filter used throughout the original pipeline
    (`Load_multiple_cycles.butter_bandpass_filter`); see the paper Section
    III-A for the upstream 15-400 Hz / 2000 Hz preprocessing applied before
    cycle segmentation, which is a separate step not performed by this class.
    """

    def __init__(self, lowcut: float = 25.0, highcut: float = 400.0, fs: float = 1000.0, order: int = 4):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self._b, self._a = self._design()

    def _design(self):
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        return butter(self.order, [low, high], btype='band')

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return filtfilt(self._b, self._a, signal)


class CycleTensorBuilder:
    """Stacks `n_cycles` consecutive, non-overlapping cardiac cycles from a
    record into one `(n_cycles, cycle_length)` tensor (Section III-B of the
    paper). Equivalent to `Load_multiple_cycles.retun_multiple_cycles`
    (`build`) and `prepare_data` (`build_from_record`).
    """

    def __init__(self, n_cycles: int):
        self.n_cycles = n_cycles

    def build(self, cycles: np.ndarray):
        """Group `cycles` (shape `(num_cycles, cycle_length)`) into tensors of
        `n_cycles` rows each. Returns `(flat, tensor)` where `tensor` has
        shape `(num_tensors, n_cycles, cycle_length)` and `flat` is the same
        data reshaped to `(num_tensors, n_cycles * cycle_length)`, or
        `(None, None)` if there are not more than `n_cycles` cycles available
        (matches the original's strict `len(x) > n` check -- a record with
        exactly `n` cycles is dropped, not just one with fewer).
        """
        n = self.n_cycles
        if len(cycles) > n and cycles.ndim == 2:
            num_full_groups = (cycles.shape[0] // n) * n
            truncated = cycles[:num_full_groups, :]
            grouped = np.stack(
                [truncated[i:i + n, :] for i in range(0, truncated.shape[0], n)],
                axis=0,
            )
            flat = np.reshape(grouped, (-1, grouped.shape[1] * grouped.shape[2]))
            return flat, grouped
        return None, None

    def build_from_record(self, cycles_per_record, labels_per_record, bandpass: BandpassFilter):
        """Apply `bandpass` then `build()` to every record in a `.mat` subset
        and concatenate the results across records. `cycles_per_record` /
        `labels_per_record` are the `X`/`Y` cell arrays loaded from a
        `training-<x>_noFIR.mat` file (see `data/README.md`). Equivalent to
        `Load_multiple_cycles.prepare_data`.
        """
        flat_all, tensor_all, label_all = [], [], []
        for cycles, labels in zip(cycles_per_record[0], labels_per_record[0]):
            filtered = bandpass(cycles)
            flat, tensor = self.build(filtered)
            if flat is not None and tensor is not None:
                flat_all.append(flat)
                tensor_all.append(tensor)
                label_all.append(labels[:tensor.shape[0]])
        return (
            np.concatenate(flat_all, axis=0),
            np.concatenate(tensor_all, axis=0),
            np.concatenate(label_all, axis=0),
        )
