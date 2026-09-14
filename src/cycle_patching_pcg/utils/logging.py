# --------------------------------------------------------
# Swin Transformer / FocalNet
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
#
# Thin OO wrapper around the same console+file logger set up by the
# `logger.create_logger` function this repository reconstructed for the
# flat-script release (see README.md > "Repository notes"): identical
# handler/formatter setup, now exposed as a class with `.info`/`.warning`/
# etc. methods instead of a bare `logging.Logger`.

import os
import sys
import logging
import functools


class ExperimentLogger:
    """Console + per-experiment-file logger.

    Args:
        output_dir: directory the log file is written to (created if needed).
        name: logger name, also used as part of the log file name; reusing
            the same `(output_dir, name, rank)` returns the same underlying
            logger (handlers are only attached once).
        rank: process rank in distributed training; only rank 0 logs to the
            console (all ranks still log to their own file).
    """

    def __init__(self, output_dir: str, name: str = '', rank: int = 0):
        self.output_dir = output_dir
        self.name = name
        self.rank = rank
        self._logger = self._build(output_dir, name, rank)

    @staticmethod
    @functools.lru_cache()
    def _build(output_dir, name, rank):
        os.makedirs(output_dir, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'

        if rank == 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(console_handler)

        file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{rank}.txt'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

        return logger

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def debug(self, msg):
        self._logger.debug(msg)

    def error(self, msg):
        self._logger.error(msg)
