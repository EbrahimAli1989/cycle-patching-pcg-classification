"""Cycle-patching, domain-balanced training with tracking (DBTT), and a
dual-branch focal modulation transformer for phonocardiogram (PCG)
classification.

Companion code for: E. A. Nehary and S. Rajan, "Cycle-Patching and
Domain-Balance with Tracking for Phonocardiogram Classification," IEEE Open
Journal of Instrumentation and Measurement (in minor revision).

See README.md for the method summary and `scripts/train.py` for a runnable
end-to-end example wiring these classes together.
"""

from .config import get_config
from .data import (
    BandpassFilter,
    CentralizedTrainingDataset,
    CycleTensorBuilder,
    CycleTensorDataset,
    DomainBalancedBatchDataset,
    DomainTrackingDataset,
    PhysioNetDataModule,
)
from .metrics import ClassificationReport, MetricsCalculator
from .models import Signals_FocalNet
from .training import RecordEvaluator, Trainer
from .utils import CheckpointManager, ExperimentLogger

__version__ = '0.1.0'

__all__ = [
    'get_config',
    'PhysioNetDataModule',
    'BandpassFilter',
    'CycleTensorBuilder',
    'CycleTensorDataset',
    'CentralizedTrainingDataset',
    'DomainBalancedBatchDataset',
    'DomainTrackingDataset',
    'Signals_FocalNet',
    'Trainer',
    'RecordEvaluator',
    'MetricsCalculator',
    'ClassificationReport',
    'ExperimentLogger',
    'CheckpointManager',
]
