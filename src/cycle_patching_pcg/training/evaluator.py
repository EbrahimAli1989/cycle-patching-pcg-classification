"""Record-level evaluation via majority voting across every cycle-tensor
built from a record.

Equivalent to `evalute_vote()` from the original flat `train.py`, turned
into a class holding the model/preprocessing objects instead of re-passing
them on every call, plus a small `evaluate_all` convenience wrapper for the
per-subset loop that used to live in `train.py`'s evaluation branch.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

from ..data.preprocessing import BandpassFilter, CycleTensorBuilder
from ..metrics import ClassificationReport, MetricsCalculator


class RecordEvaluator:
    """Classifies every cycle-tensor built from a record independently, then
    assigns the record the majority-voted label (Section III-D "Majority
    voting" in the paper).

    Args:
        model: a trained `Signals_FocalNet` in `eval()` mode.
        cycle_builder: a `CycleTensorBuilder` with the same `n_cycles` used
            for training (`--cycleno`).
        bandpass: a `BandpassFilter` (defaults match training).
        device: `torch.device` the model lives on.
    """

    def __init__(self, model, cycle_builder: CycleTensorBuilder, bandpass: BandpassFilter, device):
        self.model = model
        self.cycle_builder = cycle_builder
        self.bandpass = bandpass
        self.device = device

    @torch.no_grad()
    def evaluate_subset(self, mat_data: dict, subset_name: str = 'A') -> tuple[ClassificationReport, ClassificationReport]:
        """Evaluates one loaded `training-val_<x>_noFIR.mat` subset.

        Returns `(cycle_report, record_report)`:
          - `cycle_report`: metrics over every individual cycle-tensor
            prediction (each inheriting its record's label).
          - `record_report`: metrics over the majority-voted, record-level
            predictions (the paper's headline numbers).
        """
        X, Y = mat_data['X'], mat_data['Y']
        y_true, y_pred = [], []
        cycle_y_true, cycle_y_pred = [], []

        self.model.eval()
        for cycles, labels in zip(X[0], Y[0]):
            filtered = self.bandpass(cycles)
            _, tensor = self.cycle_builder.build(filtered)
            if tensor is None:
                continue

            images = torch.tensor(tensor).unsqueeze(dim=1).float().to(self.device)
            output = self.model(images)

            y_true.append(labels[0])
            max_indices = torch.argmax(output, dim=1)
            cycle_y_true.extend(np.repeat(labels[0], images.shape[0], axis=0))
            cycle_y_pred.extend(max_indices)

            votes = torch.bincount(max_indices)
            y_pred.append(torch.argmax(votes).item())

        print('-----------------------------------------------------')
        print(subset_name)
        print('-----------------------------------------------------')
        cm_record = confusion_matrix(y_true=y_true, y_pred=y_pred)
        print(cm_record)
        print(classification_report(y_true=y_true, y_pred=y_pred))

        record_report = MetricsCalculator.from_confusion_matrix(cm_record)
        cm_cycle = confusion_matrix(y_true=np.array(cycle_y_true), y_pred=torch.stack(cycle_y_pred).cpu().numpy())
        cycle_report = MetricsCalculator.from_confusion_matrix(cm_cycle)

        return cycle_report, record_report

    def evaluate_all(self, subset_to_mat: dict) -> dict:
        """Runs `evaluate_subset` over every `{name: mat_data}` entry (e.g.
        the dict returned by `PhysioNetDataModule.load_official_test_set()`)
        and returns `{name: (cycle_report, record_report)}`.
        """
        return {
            name: self.evaluate_subset(mat_data, subset_name=name.upper())
            for name, mat_data in subset_to_mat.items()
        }
