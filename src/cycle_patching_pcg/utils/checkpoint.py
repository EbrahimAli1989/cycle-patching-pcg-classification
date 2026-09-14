# --------------------------------------------------------
# Swin Transformer / FocalNet
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
#
# OO wrapper around the same save/load/auto-resume logic as the
# `custom_utils.py` module this repository reconstructed for the flat-script
# release (see README.md > "Repository notes"), now scoped to one output
# directory instead of taking `config.OUTPUT` on every call.

import os

import torch


class CheckpointManager:
    """Saves/loads/auto-resumes checkpoints for one experiment's output
    directory (`<output>/<modelname>/<tag>`, see `config.py`).
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def path_for_epoch(self, epoch: int) -> str:
        return os.path.join(self.output_dir, f'ckpt_epoch_{epoch}.pth')

    def save(self, epoch, model, optimizer, lr_scheduler, max_accuracy, logger, config=None):
        save_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'max_accuracy': max_accuracy,
            'epoch': epoch,
        }
        if config is not None:
            save_state['config'] = config

        save_path = self.path_for_epoch(epoch)
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")

    def load(self, config, model, optimizer, lr_scheduler, logger):
        """Loads `config.MODEL.RESUME` into `model` (and, unless
        `config.EVAL_MODE`, restores optimizer/scheduler state and advances
        `config.TRAIN.START_EPOCH`). Returns the checkpoint's `max_accuracy`
        (0.0 if absent). Mutates `config` in place, matching the original
        `custom_utils.load_checkpoint`.
        """
        logger.info(f"==============> Resuming from {config.MODEL.RESUME} ....................")
        if config.MODEL.RESUME.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.MODEL.RESUME, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=False)
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(msg)

        max_accuracy = 0.0
        if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            config.defrost()
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
            config.freeze()
            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            if 'max_accuracy' in checkpoint:
                max_accuracy = checkpoint['max_accuracy']

        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return max_accuracy

    def auto_resume(self):
        """Returns the path to the most recently modified `*.pth` checkpoint
        in `self.output_dir`, or `None` if there isn't one.
        """
        if not os.path.isdir(self.output_dir):
            return None
        checkpoints = [ckpt for ckpt in os.listdir(self.output_dir) if ckpt.endswith('.pth')]
        print(f"All checkpoints found in {self.output_dir}: {checkpoints}")
        if len(checkpoints) > 0:
            latest = max([os.path.join(self.output_dir, d) for d in checkpoints], key=os.path.getmtime)
            print(f"The latest checkpoint found: {latest}")
            return latest
        return None
