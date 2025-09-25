# encoding: utf-8
"""
credit:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py
"""

import logging
import time
import weakref
from typing import Dict

import numpy as np
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from ..utils import comm
from ..utils.events import EventStorage, get_event_storage
from ..utils.params import ContiguousParams

__all__ = ["HookBase"]

logger = logging.getLogger(__name__)


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    Each hook can implement 6 methods. The way they are called is demonstrated
    in the following snippet:
    .. code-block:: python
        hook.before_train()
        for _ in range(start_epoch, max_epoch):
            hook.before_epoch()
            for iter in range(start_iter, max_iter):
                hook.before_step()
                trainer.run_step()
                hook.after_step()
            hook.after_epoch()
        hook.after_train()
    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).
        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.
           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.
    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_epoch(self):
        """
        Called before each epoch.
        """
        pass

    def after_epoch(self):
        """
        Called after each epoch.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass
