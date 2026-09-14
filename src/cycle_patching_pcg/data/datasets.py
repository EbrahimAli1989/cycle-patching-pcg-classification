"""PyTorch `Dataset` classes for the three training modes compared in the
paper, plus a plain validation/test dataset. Logic is ported unchanged from
`Load_multiple_cycles.py` (`CustomDataset`, `CustomDataset_training`,
`CustomDataset_pytorch_V2`, `BalancedDataGeneratorWithTracking`); only the
names and docstrings have been made clearer for public release. See
README.md > "Repository notes".
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset


class CycleTensorDataset(Dataset):
    """Plain `(tensor, label)` dataset with no batching logic of its own.
    Used for the validation/held-out-test dataloaders (equivalent to
    `Load_multiple_cycles.CustomDataset`); the channel dimension is added
    downstream by `Trainer.validate`, matching the original script.
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CentralizedTrainingDataset(Dataset):
    """Training dataset for the **centralized training** baseline
    (`--training_type nothing`): all source subsets concatenated and
    shuffled with no per-domain balancing. Equivalent to
    `Load_multiple_cycles.CustomDataset_training`. The `unsqueeze(dim=1)`
    below is preserved exactly from the original (not "fixed" to `dim=0`)
    since `Trainer.train_one_epoch`'s permute logic for this training mode
    expects it.
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.tensor(self.data[idx]).unsqueeze(dim=1).float()
        return img, self.labels[idx]


class DomainBalancedBatchDataset(Dataset):
    """Training dataset for the **domain-balanced training** baseline
    (`--training_type balance`): every `__getitem__` call returns one full
    batch with an equal number of normal/abnormal tensors from every source
    dataset (no usage tracking/prioritization -- that's
    `DomainTrackingDataset`, the proposed method). Wrap with
    `DataLoader(batch_size=1)` since each item is already a batch.
    Equivalent to `Load_multiple_cycles.CustomDataset_pytorch_V2`.

    Note: `n_samples` (not `batch_size`) is what actually determines samples
    drawn per domain per batch here (`n_samples // 2` per class); `batch_size`
    is accepted for interface parity with `DomainTrackingDataset` and only
    used in `__len__`. This mirrors the original class exactly -- see
    `PhysioNetDataModule.train_dataloader` for the literal `n_samples`/
    `batch_size` values used to reproduce the paper's results.
    """

    def __init__(self, data, labels, batch_size, n_samples):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.groups = list(data.keys())
        self._reshuffle()

    def __len__(self):
        min_group_size = min(len(self.data[group]) for group in self.groups)
        return int(min_group_size // self.n_samples)

    def __getitem__(self, index):
        X, y = self._generate_batch()
        return X.clone().detach(), y.clone().detach()

    def _reshuffle(self):
        for group in self.groups:
            indices = torch.randperm(len(self.data[group]))
            self.data[group] = self.data[group][indices]
            self.labels[group] = self.labels[group][indices]

    def _generate_batch(self):
        X_batch, y_batch = [], []
        for group in self.groups:
            group_data = self.data[group]
            group_labels = self.labels[group]

            class_0_indices = np.where(group_labels == 0)[0]
            class_1_indices = np.where(group_labels == 1)[0]

            num_samples_per_class = self.n_samples // 2
            selected_0 = class_0_indices[torch.randperm(len(class_0_indices))[:num_samples_per_class]]
            selected_1 = class_1_indices[torch.randperm(len(class_1_indices))[:num_samples_per_class]]

            selected = np.concatenate((selected_0, selected_1), axis=0)
            X_batch.append(group_data[selected])
            y_batch.append(group_labels[selected])

        X_batch = torch.tensor(np.concatenate(X_batch, axis=0))
        y_batch = torch.tensor(np.concatenate(y_batch, axis=0)).long()
        return X_batch, y_batch


class DomainTrackingDataset(Dataset):
    """Training dataset for **domain-balanced training with tracking
    (DBTT)** -- the paper's proposed method (`--training_type tracking`).
    At every batch, draws an equal number of normal/abnormal tensors from
    each source dataset, preferring tensors used least often so far (a
    per-tensor visit counter incremented on every draw). This is Algorithm 1
    / Eq. (1)-(3) in the paper. Wrap with `DataLoader(batch_size=1)` since
    each item is already a batch. Equivalent to
    `Load_multiple_cycles.BalancedDataGeneratorWithTracking`.
    """

    def __init__(self, train_data, train_labels, batch_size, autoencoder=False):
        self.train_data = train_data
        self.train_labels = train_labels
        self.batch_size = batch_size
        self.classes = [0, 1]
        self.keys = list(train_data.keys())
        self.indices = self._create_indices()
        self.visit_counts = self._initialize_visit_counts()
        self.autoencoder = autoencoder

    def _create_indices(self):
        indices = {key: {cls: [] for cls in self.classes} for key in self.keys}
        for key in self.keys:
            _, y = self.train_data[key], self.train_labels[key]
            for i, label in enumerate(y):
                indices[key][label].append(i)
        return indices

    def _initialize_visit_counts(self):
        return {key: np.zeros(len(self.train_data[key]), dtype=int) for key in self.keys}

    def __len__(self):
        total_samples = sum(len(self.train_data[key]) for key in self.keys)
        return int(np.floor(total_samples / self.batch_size))

    def __getitem__(self, index):
        batch_indices = []
        samples_per_subset = self.batch_size // len(self.keys)
        samples_per_class = samples_per_subset // len(self.classes)

        for key in self.keys:
            for cls in self.classes:
                available_indices = self.indices[key][cls]
                visit_counts = self.visit_counts[key][available_indices]
                sorted_indices = [x for _, x in sorted(zip(visit_counts, available_indices))]
                chosen_indices = sorted_indices[:samples_per_class]
                batch_indices.extend((key, idx) for idx in chosen_indices)
                self.visit_counts[key][chosen_indices] += 1

        batch_data = [self.train_data[key][i] for key, i in batch_indices]
        batch_labels = [self.train_labels[key][i] for key, i in batch_indices]

        batch_data, batch_labels = shuffle(np.array(batch_data), np.array(batch_labels))

        batch_data = torch.tensor(batch_data, dtype=torch.float32)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        if self.autoencoder:
            return batch_data, batch_data
        return batch_data, batch_labels
