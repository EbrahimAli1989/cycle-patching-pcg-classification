#!/usr/bin/env python
"""Train and evaluate the cycle-patching PCG classifier.

Thin CLI orchestration layer over the `cycle_patching_pcg` package -- wires
together `PhysioNetDataModule`, `Signals_FocalNet`, `Trainer` and
`RecordEvaluator` the same way the original flat `train.py`'s `main()` did
(same CLI flags, same control flow, same hardcoded architecture arguments),
just moved out of module-level functions and into reusable classes. See
README.md > "Repository notes" for the full list of what changed vs. the
original research scripts.

Usage:
    python scripts/train.py --modelname pcg_dbtt --training_type tracking
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from cycle_patching_pcg.config import get_config  # noqa: E402
from cycle_patching_pcg.data import PhysioNetDataModule, BandpassFilter, CycleTensorBuilder  # noqa: E402
from cycle_patching_pcg.models import Signals_FocalNet  # noqa: E402
from cycle_patching_pcg.optim import build_optimizer, build_scheduler  # noqa: E402
from cycle_patching_pcg.training import Trainer, RecordEvaluator  # noqa: E402
from cycle_patching_pcg.utils import CheckpointManager, ExperimentLogger  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Cycle-patching PCG training and evaluation script')
    parser.add_argument('--cfg', type=str, default='configs/Focal_options.yaml', metavar='FILE',
                         help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY VALUE' pairs.",
                         default=None, nargs='+')

    parser.add_argument('--batch-size', type=int, help='batch size for a single GPU')
    parser.add_argument('--dataset', type=str, default='imagenet', help='dataset name')
    parser.add_argument('--data-path', type=str, help='path to dataset (default: <repo_root>/data)')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'])
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help='gradient accumulation steps')
    parser.add_argument('--use-checkpoint', action='store_true',
                         help='whether to use gradient checkpointing to save memory')
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'])
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                         help='root of output folder; full path is <output>/<modelname>/<tag>')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='test throughput only')
    parser.add_argument('--debug', action='store_true', help='perform debug only')
    parser.add_argument('--modelname', default='H_V_small', type=str, help='model/run name')
    parser.add_argument('--trainset', default='all', type=str,
                         help='[train sets that includes a, b, c, d, e, f, all] (currently unused, kept for CLI parity)')
    parser.add_argument('--withoutliers', default='y', type=str,
                         help='[with outliers: --> y, without outliers --> n] (currently unused, kept for CLI parity)')
    parser.add_argument('--flodno', default=0, type=int,
                         help='[fold number from 0 to 4] (currently unused, kept for CLI parity)')
    parser.add_argument('--cycleno', default=6, type=int, help='number of cycles per constructed tensor (N)')
    parser.add_argument('--training_type', default='balance', type=str,
                         help='[balance, nothing, tracking] -- see README.md "Training"')
    parser.add_argument('--row_w', default=100, type=int, help='horizontal patch width')
    parser.add_argument('--row_h', default=3, type=int, help='horizontal patch height')
    parser.add_argument('--col_w', default=3, type=int, help='vertical patch width')
    parser.add_argument('--col_h', default=3, type=int, help='vertical patch height')
    parser.add_argument('--random_state', default=42, type=int, help='seed for the train/validation split')
    parser.add_argument('--not_include_e', action='store_true', help='drop subset e from training')
    parser.add_argument('--save_results', type=str, default='r1', help='results CSV filename (without extension)')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for DistributedDataParallel')
    return parser


def build_criterion(config):
    if config.AUG.MIXUP > 0.:
        return SoftTargetCrossEntropy()
    if config.MODEL.LABEL_SMOOTHING > 0.:
        return LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    return torch.nn.CrossEntropyLoss()


def scale_learning_rates(config):
    """Linearly scales BASE_LR/WARMUP_LR/MIN_LR by (batch_size / 512), and by
    ACCUMULATION_STEPS when gradient accumulation is used. Unchanged from
    the original __main__ block (may not be optimal, kept as-is).
    """
    scale = config.DATA.BATCH_SIZE / 512.0
    accumulation = config.TRAIN.ACCUMULATION_STEPS if config.TRAIN.ACCUMULATION_STEPS > 1 else 1

    config.defrost()
    config.TRAIN.BASE_LR = config.TRAIN.BASE_LR * scale * accumulation
    config.TRAIN.WARMUP_LR = config.TRAIN.WARMUP_LR * scale * accumulation
    config.TRAIN.MIN_LR = config.TRAIN.MIN_LR * scale * accumulation
    config.OUTPUT = os.getenv('PT_OUTPUT_DIR') or config.OUTPUT
    config.freeze()
    return config


def main():
    args = build_arg_parser().parse_args()
    config = get_config(args)
    config = scale_learning_rates(config)

    # NOTE: distributed training (torch.distributed / DDP) is present but
    # commented out in the original scripts this repository is based on and
    # is not wired up here either; only single-process/single-GPU training
    # is supported. `seed = 0` (not `config.SEED + rank`) is preserved as-is.
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = ExperimentLogger(output_dir=config.OUTPUT, name=args.modelname)
    logger.info(config.dump())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Creating model: {config.MODEL.TYPE}/{args.modelname}")
    # NOTE: img_h is fixed at 12 here regardless of --cycleno; the number of
    # rows in the input tensor actually constructed by PhysioNetDataModule is
    # controlled by --cycleno (default 6). If you change --cycleno from the
    # value used to produce your checkpoints, update img_h to match (it must
    # equal the tensor's cycle count N), or the patch embedding's input
    # resolution will be wrong. Left as in the original ablation script
    # rather than silently rewired -- see README.md > "Repository notes".
    model = Signals_FocalNet(
        depths_h=[2], depths_v=[2], embed_dim=96,
        focal_levels_h=[3], focal_levels_v=[3],
        row_h=args.row_h, row_w=args.row_w, col_h=args.col_h, col_w=args.col_w,
        img_h=12, img_w=2500, in_chans=1, num_classes=2, col_or_row='all',
    ).to(device)
    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    optimizer = build_optimizer(config, model)
    criterion = build_criterion(config)

    data_path = args.data_path or os.path.join(_REPO_ROOT, 'data')
    data_module = PhysioNetDataModule(
        data_path=data_path,
        n_cycles=args.cycleno,
        random_state=args.random_state,
        not_include_e=args.not_include_e,
    )
    if not config.DEBUG_MODE:
        data_module.setup()
        train_loader = data_module.train_dataloader(args.training_type)
        val_loaders = data_module.val_dataloaders()
    else:
        train_loader, val_loaders = None, {}

    lr_scheduler = build_scheduler(config, optimizer, len(train_loader)) if train_loader is not None else None
    checkpoint_manager = CheckpointManager(config.OUTPUT)
    trainer = Trainer(config, model, optimizer, lr_scheduler, criterion, device, logger, checkpoint_manager)

    if not config.DEBUG_MODE:
        trainer.run_training_phase(train_loader, val_loaders, args.training_type)

    # Evaluation phase: record-level majority voting over the official
    # PhysioNet validation subsets a-e, matching the original main(training=False).
    logger.info('Start testing model.......')
    ckpt_path = checkpoint_manager.path_for_epoch(config.TRAIN.EPOCHS - 1)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    evaluator = RecordEvaluator(
        model=model,
        cycle_builder=CycleTensorBuilder(args.cycleno),
        bandpass=BandpassFilter(),
        device=device,
    )
    test_data = data_module.load_official_test_set()
    results = evaluator.evaluate_all(test_data)

    ordered = ['a', 'b', 'c', 'd', 'e']
    cycle_rows = [results[name][0].as_row() for name in ordered]
    record_rows = [results[name][1].as_row() for name in ordered]

    print('=============================================')
    print('cycle results...........')
    print(np.array(cycle_rows))
    print('record results.........')
    print(np.array(record_rows))

    df = pd.DataFrame(record_rows, columns=results['a'][1].columns())
    df.to_csv(args.save_results + '.csv', index=False)


if __name__ == '__main__':
    main()
