import logging
import sys
import os

import torch
import torch.distributed as dist


def setup_process(rank, world_size, master_port='12355'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_process():
    dist.destroy_process_group()


def save_training_data(path, optimizer=None, scaler=None, epoch=None):
    checkpoint = {
        'optimizer': None if optimizer is None else optimizer.state_dict(),
        'scaler': None if scaler is None else scaler.state_dict(),
        'epoch': epoch
    }

    torch.save(checkpoint, os.path.join(path, 'training_data.pt'))


def load_training_data(path, optimizer=None, scaler=None, map_location=None):
    checkpoint = torch.load(os.path.join(path, 'training_data.pt'), map_location=map_location)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])

    return checkpoint


class Logger:
    def __init__(self, log_dir=None, enabled=True, pad_length=50):
        self._logger = self._get_logger(log_dir) if enabled else None
        self._pad_length = pad_length

    def _pad_message(self, message):
        return (" " + message + " ").center(self._pad_length, '=')

    def info(self, message, pad=False):
        if self._logger is not None:
            message = self._pad_message(message) if pad else message
            self._logger.info(message)

    def line(self):
        if self._logger is not None:
            self._logger.info('=' * self._pad_length)

    @staticmethod
    def _get_logger(log_dir=None):
        """
        get a logger for displaying information to console or log to file (optional)
        :param log_dir: str, logging path. None for not log to file
        :return: logger
        """
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.flush = sys.stdout.flush
        logger.addHandler(stream_handler)

        if log_dir is not None:
            file_handler = logging.FileHandler(log_dir)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


class TaskType:
    AFTER = 'after'
    BEFORE = 'before'
    INTENT = 'intent'
    CAPTION = 'caption'
    REGION_CAPTION = 'region_caption'

    ALL_TYPES = {AFTER, BEFORE, INTENT, CAPTION, REGION_CAPTION}
