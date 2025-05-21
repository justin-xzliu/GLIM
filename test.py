import os
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import torch
import warnings
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from model.glim import GLIM
from data.datamodule import GLIMDataModule

warnings.filterwarnings("ignore", ".*when logging on epoch level in distributed.*")
# set logger
group_name = 'dev-dist-test'
log_dir = './runs/' + group_name
os.makedirs(log_dir, exist_ok=True)

devices = [0]
L.seed_everything(42, workers=True)
torch.set_float32_matmul_precision('medium')

# logger = None
logger = WandbLogger(project = 'glim',
                     group = group_name,
                     save_dir = log_dir,
                    #  version = 'w25zyxe5',
                    #  offline = True,
                    )

trainer = L.Trainer(accelerator = 'gpu',
                    devices = devices,
                    logger = logger,
                    precision = 'bf16-mixed', 
                    # limit_test_batches=2
                    )

dm = GLIMDataModule(data_path = './data/tmp/zuco_eeg_label_8variants.df',
                    eval_noise_input = False,
                    bsz_test = 24,
                    num_workers = 2,
                    )

model = GLIM.load_from_checkpoint(
    "checkpoints/glim-zuco-epoch=199-step=49600.ckpt",
    map_location = f"cuda:{devices[0]}",
    strict = False,
    # evaluate_prompt_embed = 'zero',
    # prompt_dropout_probs = (0.0, 0.1, 0.1),
    )
trainer.test(model, datamodule=dm)