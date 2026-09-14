# Cycle-Patching and Domain-Balance with Tracking for Phonocardiogram Classification

Code accompanying:

> E. A. Nehary and S. Rajan, "Cycle-Patching and Domain-Balance with Tracking
> for Phonocardiogram Classification," *IEEE Open Journal of Instrumentation
> and Measurement* (in minor revision).

An installable, object-oriented PyTorch package implementing a
focal-modulation-transformer classifier for normal/abnormal heart sound
(phonocardiogram, PCG) classification that:

- builds **multi-cycle input tensors** (`N` consecutive cardiac cycles
  stacked as an `N × L` matrix) instead of single cycles, to dampen the
  effect of noisy, record-inherited cycle labels;
- extracts features with two **focal modulation transformer** branches fed
  with **horizontal** and **vertical patches** of that tensor (individual
  heart-sound morphology vs. cross-cycle correlation), then fuses and
  classifies;
- trains with **domain-balanced training with tracking (DBTT)**: a pointer
  mechanism that prioritizes samples from each source dataset that have been
  used least often, so heterogeneous, unequally-sized datasets all
  contribute fairly to training;
- evaluates at the record level with **majority voting** across every
  constructed tensor belonging to a record.

Trained and evaluated on the public **2016 PhysioNet/CinC Challenge** heart
sound database (subsets a-f, six sensors/sites/countries).

## Package layout

```
├── pyproject.toml               # pip install -e . installs the package below
├── scripts/train.py             # CLI entry point: wires the classes together, trains + evaluates
├── configs/Focal_options.yaml   # default training config (epochs, LR, optimizer, ...)
├── data/README.md               # expected data layout + how to get the PhysioNet 2016 dataset
└── src/cycle_patching_pcg/
    ├── config.py                 # yacs-based config schema + CLI/YAML merge (adapted from Microsoft FocalNet)
    ├── metrics.py                 # MetricsCalculator / ClassificationReport (Recall, Specificity, F1, MCC, Kappa, ...)
    ├── models/
    │   └── focal_net.py           # Signals_FocalNet: dual-branch focal modulation transformer
    ├── data/
    │   ├── preprocessing.py       # BandpassFilter, CycleTensorBuilder
    │   ├── datasets.py            # CycleTensorDataset, CentralizedTrainingDataset, DomainBalancedBatchDataset, DomainTrackingDataset
    │   └── datamodule.py          # PhysioNetDataModule: loads .mat files, builds dataloaders for all 3 training modes
    ├── training/
    │   ├── trainer.py             # Trainer: training loop, validation, checkpointing, resume
    │   └── evaluator.py           # RecordEvaluator: record-level majority-vote evaluation
    ├── optim/
    │   ├── optimizer.py           # build_optimizer (adapted from Microsoft FocalNet)
    │   └── scheduler.py           # build_scheduler (adapted from Microsoft FocalNet)
    └── utils/
        ├── logging.py             # ExperimentLogger
        ├── checkpoint.py          # CheckpointManager
        └── training_utils.py      # get_grad_norm, reduce_tensor
```

## Method summary

1. **Preprocessing** (done upstream of this repo, see `data/README.md`): PCG
   recordings are bandpass filtered (3rd-order Butterworth, 15-400 Hz),
   resampled to 2000 Hz, segmented into individual cardiac cycles, then
   z-score normalized and zero-padded to a fixed 2-second cycle length.
2. **Tensor construction** (`data.preprocessing.CycleTensorBuilder`): every
   `N` consecutive, non-overlapping cycles from a record are stacked into
   one `N × L` tensor; a tensor inherits its label from the record. `N` is
   `--cycleno` (paper explores `N ∈ {3,5,7,9,11,13}`, see Section V of the
   paper for the ablation).
3. **Horizontal / vertical patching + Focal Modulation Transformer**
   (`models.focal_net.Signals_FocalNet`): the tensor is patch-embedded
   twice, once with wide/short patches (`--row_w 100 --row_h 3`,
   individual-cycle features) and once with small square patches
   (`--col_w 3 --col_h 3`, inter-cycle correlation), each fed through its
   own focal modulation transformer branch (depth 2, embed dim 96, focal
   level 3, focal window 3). The two pooled feature vectors are concatenated
   and passed to a linear classification head.
4. **Domain-balanced training with tracking**
   (`data.datasets.DomainTrackingDataset`): at every training step, an equal
   number of normal/abnormal tensors is drawn from *each* source dataset,
   preferring tensors with the lowest usage count so far (a per-tensor
   pointer incremented every time it's drawn) -- this is Algorithm 1 /
   Eq. (1)-(3) in the paper.
5. **Record-level evaluation via majority voting**
   (`training.evaluator.RecordEvaluator`): every tensor built from a test
   record is classified independently; the record's predicted label is the
   majority vote (mode) over those predictions.
   `metrics.MetricsCalculator` then computes Recall, Specificity, Precision,
   F1, Accuracy, Matthews Correlation Coefficient and Kappa, both
   per-cycle-tensor and per-record.



Requires Python ≥3.9 and a PyTorch build matching your hardware -- see
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
for the right install command for your GPU/CUDA version, or just
`pip install -e .` on CPU-only PyTorch to start. A GPU is strongly
recommended for real runs -- the paper trains for 500 epochs per
configuration, repeated across folds/seeds for the ablation studies.


## Training

`scripts/train.py` is the CLI entry point; it builds a `PhysioNetDataModule`,
a `Signals_FocalNet`, and a `Trainer`/`RecordEvaluator`, and runs training
followed by record-level evaluation. The three training modes compared in
the paper are switched with `--training_type`:

| `--training_type` | Method (paper terminology)              | Dataset class used |
|---|---|---|
| `nothing`          | Centralized training (baseline)         | `CentralizedTrainingDataset` -- all subsets concatenated, plain shuffle |
| `balance`           | Domain-balanced training (baseline)     | `DomainBalancedBatchDataset` -- equal samples per subset/class per batch |
| `tracking`         | **Domain-balanced training with tracking (proposed)** | `DomainTrackingDataset` -- as above, plus least-used-first pointer |

Reproduce the paper's proposed method:

```bash
python scripts/train.py \
  --modelname pcg_dbtt \
  --data-path data \
  --output output \
  --training_type tracking \
  --cycleno 6 \
  --row_h 3 --row_w 100 \
  --col_h 3 --col_w 3 \
  --withoutliers y \
  --random_state 42 \
  --save_results results_dbtt
```

This trains for `TRAIN.EPOCHS` epochs (500 by default,
`configs/Focal_options.yaml`) with AdamW and a cosine LR schedule,
checkpointing every `SAVE_FREQ` epochs (default 20) to
`output/<modelname>/default/`, then automatically runs record-level
evaluation (majority voting over subsets a-e) and writes per-fold metrics to
`<save_results>.csv` (columns: `Recall, Specificity, Precision, F1_score,
Accuracy, MatthewsCorrelationCoefficient, Kappa`).






## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{nehary2026cyclepatching,
  title   = {Cycle-Patching and Domain-Balance with Tracking for Phonocardiogram Classification},
  author  = {Nehary, Ebrahim A. and Rajan, Sreeraman},
  journal = {IEEE Open Journal of Instrumentation and Measurement},
  year    = {2026},
  note    ={Under review}
}
```
*(DOI/volume/page numbers to be added once the paper is published -- see `CITATION.cff`.)*

## Acknowledgments

- This work was supported by the Natural Sciences and Engineering Research
  Council (NSERC) of Canada.
- `models/focal_net.py`, `config.py`, `optim/scheduler.py` and
  `optim/optimizer.py` are adapted from Microsoft's
  [FocalNet](https://github.com/microsoft/FocalNet) and
  [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
  (MIT License); see [`LICENSE`](LICENSE) for details.
- Dataset: [2016 PhysioNet/CinC Challenge](https://physionet.org/content/challenge-2016/1.0.0/)
  heart sound database.

## License

MIT License, see [`LICENSE`](LICENSE) (includes third-party attribution for
the Microsoft-derived files above).
