# --------------------------------------------------------
# Focal Modulation Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com) based on Swin Transformer code
# --------------------------------------------------------
#
# `Trainer` is an OO reorganization of `train_one_epoch()`/`validate()`/
# `main(training=True)` from the original flat `train.py`. Logic (permute
# branching by training type, gradient accumulation/clipping, AMP hook,
# resume/auto-resume/checkpointing) is preserved unchanged -- only turned
# into methods on a class holding the model/optimizer/scheduler/etc. instead
# of module-level functions threading them through every call. See
# README.md > "Repository notes".

from __future__ import annotations

import datetime
import time

import torch
from timm.utils import AverageMeter, accuracy

from ..utils.training_utils import get_grad_norm

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


class Trainer:
    """Runs the training loop, validation, checkpointing and resume logic for
    one experiment.

    Args:
        config: the experiment's `yacs` `CfgNode` (see `config.py`).
        model: the `Signals_FocalNet` instance (already moved to `device`).
        optimizer / lr_scheduler: built with `cycle_patching_pcg.optim`.
        criterion: loss function (`CrossEntropyLoss` or a `timm` label-
            smoothing/soft-target variant, selected by the caller based on
            `config.MODEL.LABEL_SMOOTHING` / `config.AUG.MIXUP`).
        device: `torch.device`.
        logger: an `ExperimentLogger`.
        checkpoint_manager: a `CheckpointManager` scoped to `config.OUTPUT`.
    """

    def __init__(self, config, model, optimizer, lr_scheduler, criterion, device, logger, checkpoint_manager):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager

    def maybe_resume(self, val_loaders: dict) -> float:
        """Auto-resumes from the latest checkpoint in `config.OUTPUT` if
        `config.TRAIN.AUTO_RESUME` is set, then loads `config.MODEL.RESUME`
        (if any) and reports validation accuracy on each loader in
        `val_loaders`. Returns the checkpoint's `max_accuracy` (0.0 if no
        checkpoint was loaded). Equivalent to the pre-training-loop block of
        the original `main(training=True)`.
        """
        if self.config.TRAIN.AUTO_RESUME:
            resume_file = self.checkpoint_manager.auto_resume()
            if resume_file:
                if self.config.MODEL.RESUME:
                    self.logger.warning(
                        f"auto-resume changing resume file from {self.config.MODEL.RESUME} to {resume_file}")
                self.config.defrost()
                self.config.MODEL.RESUME = resume_file
                self.config.freeze()
                self.logger.info(f'auto resuming from {resume_file}')
            else:
                self.logger.info(f'no checkpoint found in {self.config.OUTPUT}, ignoring auto resume')

        max_accuracy = 0.0
        if self.config.MODEL.RESUME:
            max_accuracy = self.checkpoint_manager.load(
                self.config, self.model, self.optimizer, self.lr_scheduler, self.logger)
            report = ', '.join(
                f'{name}={self.validate(loader)[0]:.4f}%' for name, loader in val_loaders.items())
            self.logger.info(f"Accuracy of the resumed network: {report}")
        return max_accuracy

    def run_training_phase(self, train_loader, val_loaders: dict, training_type: str) -> float:
        """Resumes if applicable, then trains for the configured number of
        epochs -- unless a checkpoint was resumed AND `config.EVAL_MODE` is
        set, in which case training is skipped entirely (this early-return
        quirk is preserved exactly from the original `main()`: with no
        checkpoint to resume from, `--eval` alone does *not* skip training).
        """
        max_accuracy = self.maybe_resume(val_loaders)
        if self.config.MODEL.RESUME and self.config.EVAL_MODE:
            return max_accuracy
        return self.fit(train_loader, val_loaders, training_type, max_accuracy=max_accuracy)

    def fit(self, train_loader, val_loaders: dict, training_type: str, max_accuracy: float = 0.0) -> float:
        self.logger.info("Start training")
        start_time = time.time()
        for epoch in range(self.config.TRAIN.START_EPOCH, self.config.TRAIN.EPOCHS):
            self.train_one_epoch(train_loader, epoch, training_type)

            if epoch % self.config.SAVE_FREQ == 0 or epoch == (self.config.TRAIN.EPOCHS - 1):
                self.checkpoint_manager.save(
                    epoch, self.model, self.optimizer, self.lr_scheduler, max_accuracy, self.logger,
                    config=self.config)

            accs = {name: self.validate(loader)[0] for name, loader in val_loaders.items()}
            report = ', '.join(f'{name}={acc:.4f}%' for name, acc in accs.items())
            self.logger.info(f"Accuracy of the network on the {len(train_loader)} test batches: {report}")
            if 'a' in accs:
                max_accuracy = max(max_accuracy, accs['a'])
            self.logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        self.logger.info(f'Training time {total_time_str}')
        return max_accuracy

    def train_one_epoch(self, data_loader, epoch, training_type):
        # NOTE: the original train_one_epoch() also threaded a `mixup_fn`
        # argument through this loop, but `main()` only ever set it to
        # `None` (the mixup pipeline referenced by config.AUG.MIXUP was
        # never actually wired up), so the mixup branch was always a no-op.
        # Dropped here rather than kept as a permanently-unused parameter.
        self.model.train()
        self.optimizer.zero_grad()

        num_steps = len(data_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()

        start = time.time()
        end = time.time()
        for idx, (samples, targets) in enumerate(data_loader):
            # The 'nothing' (centralized) dataloader yields plain batches
            # (permute(0,2,1,3)); the 'balance'/'tracking' dataloaders yield
            # one pre-built batch per __getitem__ wrapped by DataLoader(
            # batch_size=1), hence the leading singleton dim removed by
            # permute(1,0,2,3) instead. Preserved exactly from the original
            # train_one_epoch().
            if training_type == 'nothing':
                samples = samples.permute(0, 2, 1, 3).float().to(self.device)
            else:
                samples = samples.permute(1, 0, 2, 3).float().to(self.device)

            targets = targets.squeeze(dim=0).long().to(self.device)

            outputs = self.model(samples)

            if self.config.TRAIN.ACCUMULATION_STEPS > 1:
                loss = self.criterion(outputs, targets)
                loss = loss / self.config.TRAIN.ACCUMULATION_STEPS
                if self.config.AMP_OPT_LEVEL != "O0":
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if self.config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer), self.config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(self.optimizer))
                else:
                    loss.backward()
                    if self.config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(self.model.parameters())
                if (idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(self.model.parameters())

                self.optimizer.step()
                self.lr_scheduler.step_update(epoch * num_steps + idx)

            loss_meter.update(loss.item(), targets.size(0))
            norm_meter.update(grad_norm)
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.config.PRINT_FREQ == 0:
                lr = self.optimizer.param_groups[0]['lr']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
                etas = batch_time.avg * (num_steps - idx)
                self.logger.info(
                    f'Train: [{epoch}/{self.config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
        epoch_time = time.time() - start
        self.logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    @torch.no_grad()
    def validate(self, data_loader):
        criterion = torch.nn.CrossEntropyLoss()
        self.model.eval()

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()

        end = time.time()
        for idx, (images, target) in enumerate(data_loader):
            images = images.unsqueeze(dim=1).float().to(self.device)
            target = target.long().to(self.device)

            output = self.model(images)
            loss = criterion(output, target)
            acc1 = accuracy(output, target)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1[0], target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
                self.logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')
        self.logger.info(f' * Acc@1 {acc1_meter.avg:.3f}')

        return acc1_meter.avg, loss_meter.avg

    @torch.no_grad()
    def throughput(self, data_loader):
        self.model.eval()
        for images, _ in data_loader:
            images = images.to(self.device, non_blocking=True)
            batch_size = images.shape[0]
            for _ in range(50):
                self.model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.logger.info("throughput averaged with 30 times")
            tic1 = time.time()
            for _ in range(30):
                self.model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            tic2 = time.time()
            self.logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
            return
