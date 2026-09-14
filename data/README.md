# Data

This repository does not ship any data. It expects the **2016 PhysioNet/CinC
Challenge** heart sound (phonocardiogram) database, filtered/segmented into
per-record cycle matrices and saved as MATLAB `.mat` files, one per public
subset (`a`-`f`) plus one per official validation subset (`val_a`-`val_e`).

## 1. Get the raw challenge data

Download the public training/validation sets from PhysioNet:
<https://physionet.org/content/challenge-2016/1.0.0/>

## 2. Expected file layout

`cycle_patching_pcg.data.PhysioNetDataModule` (used by `scripts/train.py`)
reads `.mat` files named like this from `--data-path` (default: this
`data/` folder):

```
data/
├── training-a.mat        # subset a, training portion
├── training-b.mat        # subset b
├── training-c.mat        # subset c
├── training-d.mat        # subset d
├── training-e.mat        # subset e
├── training-f.mat        # subset f
├── training-val_a_noFIR.mat    # subset a, official validation/test portion
├── training-val_b.mat
├── training-val_c.mat
├── training-val_d.mat
└── training-val_e.mat
```

Each `.mat` file must contain two cell arrays:

- `X`: one entry per record, each a `(num_cycles, cycle_length)` array of
  individually segmented cardiac cycles.
- `Y`: one entry per record, matching label(s) for that record's cycles
  (`0` = normal, `1` = abnormal).

## 3. Preprocessing pipeline (see paper, Section III-A)

Before segmentation into `X`, records were:
1. Bandpass filtered (3rd-order Butterworth, 15-400 Hz).
2. Resampled to 2000 Hz.
3. Segmented into individual cardiac cycles (Springer et al. segmentation
   algorithm, as cited in the paper).
4. Z-score normalized and zero-padded to a fixed 2-second cycle length.

This repository does not include a script that performs steps 1-4 from raw
`.wav`/challenge files -- it starts from the already-segmented `.mat` files
described above. If you only have raw PhysioNet 2016 recordings, you will
need to reproduce this preprocessing/segmentation step yourself before
`scripts/train.py` can consume the data.

## 4. Pointing the code at your data

```bash
python scripts/train.py --data-path /path/to/data --modelname my_run
```

or simply drop the `.mat` files into this `data/` folder and omit
`--data-path` (it defaults to `<repo_root>/data/`).
