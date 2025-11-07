from lightning import Callback
from torch.cuda import max_memory_allocated, reset_peak_memory_stats, synchronize

from pytorch_lightning.utilities.rank_zero import rank_zero_only
import time

from lightning.pytorch.utilities import rank_zero_only, rank_zero_info

class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        reset_peak_memory_stats(trainer.strategy.root_device.index)
        synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        synchronize(trainer.strategy.root_device.index)
        max_memory = max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass