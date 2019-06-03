""" run discriminator """
import time
import tensorflow as tf

from neuralgym.callbacks import PeriodicCallback, CallbackLoc
from neuralgym.utils.logger import ProgressBar, callback_log
from neuralgym.ops.summary_ops import scalar_summary
from trainer import Trainer


class SecondaryTrainer(PeriodicCallback, Trainer):
    """SecondaryTrainer
    This callback preiodically train discriminator for generative adversarial
    networks. Note that with this callback, the training of GAN is
    alternatively between training generator and discriminator.
    """

    def __init__(self, pstep, **context):
        PeriodicCallback.__init__(self, CallbackLoc.step_start, pstep)
        context['log_progress'] = context.pop('log_progress', False)
        Trainer.__init__(self, primary=False, **context)

    def run(self, sess, step):
        self.context['sess'] = sess
        self.train()