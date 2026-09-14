"""Loads the 2016 PhysioNet/CinC Challenge `.mat` subsets, builds N-cycle
tensors, and exposes train/validation dataloaders for all three training
modes compared in the paper.

Equivalent to `Load_multiple_cycles.returndata` + `return_data`, reorganized
as a class. The literal `n_samples`/`batch_size` constants and the
`train_test_split` validation carve-out below are preserved exactly from the
original functions -- see the docstrings for what would otherwise look like
arbitrary numbers.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .datasets import (
    CentralizedTrainingDataset,
    CycleTensorDataset,
    DomainBalancedBatchDataset,
    DomainTrackingDataset,
)
from .preprocessing import BandpassFilter, CycleTensorBuilder

TRAIN_SUBSETS = ('a', 'b', 'c', 'd', 'e', 'f')
VALIDATION_SUBSETS = ('a', 'b', 'c', 'd', 'e')  # subset f has no held-out split, as in the original


class PhysioNetDataModule:
    """Owns everything needed to go from `.mat` files on disk to dataloaders.

    Args:
        data_path: folder containing `training-<x>_noFIR.mat` /
            `training-val_<x>_noFIR.mat` files (see `data/README.md`).
        n_cycles: number of consecutive cycles per constructed tensor
            (`--cycleno` in the original CLI).
        random_state: seed for the 90/10 train/validation split carved out
            of each subset a-e (`--random_state`).
        not_include_e: drop subset `e` from the *training* dictionaries
            (subset `e` is the only one using two different sensors for
            healthy vs. patient recordings); validation dataloaders for a-e
            are unaffected, matching the original.
    """

    def __init__(self, data_path: str, n_cycles: int, random_state: int = 42, not_include_e: bool = False):
        self.data_path = self._normalize_path(data_path)
        self.n_cycles = n_cycles
        self.random_state = random_state
        self.not_include_e = not_include_e
        self.bandpass = BandpassFilter()
        self.cycle_builder = CycleTensorBuilder(n_cycles)

        self._train_tensors: dict[str, np.ndarray] | None = None
        self._train_labels: dict[str, np.ndarray] | None = None
        self._val_tensors: dict[str, np.ndarray] | None = None
        self._val_labels: dict[str, np.ndarray] | None = None

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path.endswith(os.sep) and not path.endswith('/'):
            path += os.sep
        return path

    def setup(self) -> 'PhysioNetDataModule':
        """Loads `training-<x>_noFIR.mat` for every subset a-f, builds cycle
        tensors, then carves a 10% held-out validation split out of subsets
        a-e (subset f is used entirely for training, with no split -- as in
        the original `returndata()`).
        """
        train_tensors, train_labels = {}, {}
        val_tensors, val_labels = {}, {}

        for name in TRAIN_SUBSETS:
            mat = loadmat(self.data_path + f'training-{name}_noFIR.mat')
            _, tensors, labels = self.cycle_builder.build_from_record(mat['X'], mat['Y'], self.bandpass)

            if name not in VALIDATION_SUBSETS:
                train_tensors[name] = tensors
                train_labels[name] = np.squeeze(labels, axis=1)
                continue

            tr_x, va_x, tr_y, va_y = train_test_split(
                tensors, labels, test_size=0.1, random_state=self.random_state,
            )
            train_tensors[name] = tr_x
            train_labels[name] = np.squeeze(tr_y, axis=1)
            val_tensors[name] = va_x
            val_labels[name] = np.squeeze(va_y, axis=1)

        self._train_tensors, self._train_labels = train_tensors, train_labels
        self._val_tensors, self._val_labels = val_tensors, val_labels
        return self

    def _training_subset(self):
        """Returns (keys, n_samples, batch_size) for the balanced/tracking
        loaders. These exact constants (5/30 vs. 6/36) are copied from the
        original `return_data()` -- they are not derived from `len(keys)`
        there either (with `not_include_e`, `n_samples=5` still yields 4
        samples/domain per batch via `n_samples // 2` per class -- see
        `DomainBalancedBatchDataset`), so this preserves that as-is rather
        than "fixing" it into something more internally consistent.
        """
        if self.not_include_e:
            return ['a', 'b', 'c', 'd', 'f'], 5, 30
        return ['a', 'b', 'c', 'd', 'e', 'f'], 6, 36

    def train_dataloader(self, training_type: str) -> DataLoader:
        """`training_type` is one of:
          - `'tracking'`: proposed domain-balanced training with tracking (DBTT).
          - `'balance'`: domain-balanced training baseline (no tracking).
          - anything else (e.g. `'nothing'`): centralized training baseline.
        """
        keys, n_samples, batch_size = self._training_subset()
        data = {k: self._train_tensors[k] for k in keys}
        labels = {k: self._train_labels[k] for k in keys}

        if training_type == 'balance':
            dataset = DomainBalancedBatchDataset(data, labels, batch_size=batch_size, n_samples=n_samples)
            return DataLoader(dataset, batch_size=1, shuffle=True)

        if training_type == 'tracking':
            dataset = DomainTrackingDataset(data, labels, batch_size=batch_size)
            return DataLoader(dataset, batch_size=1, shuffle=True)

        concat_data = np.concatenate([data[k] for k in keys], axis=0)
        concat_labels = np.concatenate([labels[k] for k in keys], axis=0)
        dataset = CentralizedTrainingDataset(concat_data, concat_labels)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def val_dataloaders(self) -> dict[str, DataLoader]:
        """One `DataLoader` per validation subset a-e (batch size 32, no
        shuffle), matching `dataloader_a` .. `dataloader_e` in the original.
        """
        loaders = {}
        for name in VALIDATION_SUBSETS:
            dataset = CycleTensorDataset(self._val_tensors[name], self._val_labels[name])
            loaders[name] = DataLoader(dataset, batch_size=32, shuffle=False)
        return loaders

    def load_official_test_set(self) -> dict:
        """Loads the official PhysioNet validation subsets
        (`training-val_<x>_noFIR.mat`, x in a-e) used as the held-out test
        set for the record-level majority-vote evaluation in `train.py`'s
        original evaluation branch / `RecordEvaluator`.
        """
        return {
            name: loadmat(self.data_path + f'training-val_{name}_noFIR.mat')
            for name in VALIDATION_SUBSETS
        }
