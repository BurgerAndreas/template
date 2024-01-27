import tqdm
import os
import time

from typing import Any, Dict, Union, NamedTuple, List, Optional, Callable, Tuple

import numpy as np

import hydra
import wandb
import omegaconf

import torch
import torch.nn.functional as F

from proj.model.my_model import MyModel

from proj.config.paths import DOWNLOAD_DIR, OUTPUT_DIR
from proj.dataloading.datasetloader import get_dataset
from proj.utils.checkpointing import commit_state, name_from_config, load_checkpoint

import logging


def train_loop(cfg: omegaconf.DictConfig):

    log_file = os.path.join('logs', 'train.log')
    logging.basicConfig(filename=log_file,level=logging.DEBUG)
    logging.info('\nStarting training.')

    torch.set_default_device(cfg.device)

    torch.random.manual_seed(cfg.seed)
    # snapshot of the rng before the looping that we can use to restore
    rng = torch.random.get_rng_state()

    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    criterion = hydra.utils.instantiate(cfg.criterion)

    ds = get_dataset(image_size=cfg.image_size)

    # try to load checkpoint
    model, optimizer, batch_step, curr_loss = load_checkpoint(cfg, model, optimizer)
    time_last_save = time.time()

    for i, batch in tqdm.tqdm(
        enumerate(ds.prefetch(3).batch(cfg.batch_size).as_numpy_iterator()),
        initial=batch_step,
        total=cfg.num_batches,
        desc="Training",
        unit="batch",
    ):

        optimizer.zero_grad()
        
        output = model(batch)

        loss = criterion(output, batch['label'])

        loss.backward()
        optimizer.step()

        # log
        wandb.log({"loss": loss.item()}, step=batch_step)
        logging.info(f'Batch_step={batch_step} loss: {loss.item()}')

        # VALIDATION LOOP
        # with torch.no_grad():
        #     val_loss = []
        #     for val_batch in mnist_val:
        #     x, y = val_batch
        #     logits = model(x)
        #     val_loss.append(cross_entropy_loss(logits, y).item())

        #     val_loss = torch.mean(torch.tensor(val_loss))
        #     wandb.log({"val_loss": val_loss}, step=batch_step)

        # checkpoint every xxx seconds
        if time.time() - time_last_save > cfg.checkpoint_interval:
            commit_state(cfg, model, optimizer, rng, batch_step, curr_loss)

        batch_step += 1
        # end of training
        if batch_step >= cfg.num_batches:
            break
    
    # checkpoint at the end of training
    commit_state(cfg, model, optimizer, rng, batch_step, curr_loss)

    logging.info('\nFinished training.')



if __name__ == "__main__":
    with hydra.initialize(version_base=None, config_path="../config", job_name="train"):
        cfg = hydra.compose(
            config_name="base",
            overrides=[
                "batch_size=2",
            ],
        )
        wandb.init(mode="disabled")
        os.environ["WANDB_DISABLED"] = "true"
        train_loop(cfg)
