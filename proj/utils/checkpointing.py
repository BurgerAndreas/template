import torch
import torch.nn as nn
import torch.optim as optim

import hydra
import omegaconf

import os
import sys
import time
import argparse
from datetime import datetime

import logging


from typing import Any, Dict, List, Optional, Tuple, Union

from proj.config.paths import DOWNLOAD_DIR, OUTPUT_DIR


def name_from_config(cfg: omegaconf.DictConfig) -> str:
    """Generate a name for the model based on the config.
    Name is intended to be used as a file name for saving checkpoints and outputs.
    """
    try:
        mname = cfg.name
        # override format: 'pretrain_dataset=bridge,steps=10,use_wandb=False'
        override_names = ""
        if cfg.override_dirname:
            for arg in cfg.override_dirname.split(","):
                override = arg.replace("+", "").replace("_", "")
                override = override.replace("=", "-").replace(".", "")
                override_names += "_" + override
    except Exception as error:
        print('\nname_from_config() failed:', error)
        print("cfg:", cfg)
        raise error
    logging.info("name_from_config() mname: %s, override_names: %s", mname, override_names)
    return mname + override_names


def commit_state(cfg, model, optimizer, rng, batch_step, cur_loss):
    """Save checkpoint.
    We need to be careful when saving checkpoints since preemption can also
    occur during checkpointing. Therefore, we need to make sure the checkpoint
    file is either kept untouched or successfully updated during this process.
    """
    name = name_from_config(cfg)
    checkpoint_path = os.path.join(OUTPUT_DIR, 'checkpoints')
    temp_path = os.path.join(os.path.dirname(checkpoint_path), "temp.pt")

    training_state = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        # 'sampler' : sampler.state_dict(dataloader_iter),
        'batch_step': batch_step,
        'cur_loss' : cur_loss,
        'rng' : rng
    }

    # first save to temp file
    torch.save(training_state, temp_path)
    # according to the GNU spec of rename, the state of checkpoint_path
    # is atomic, i.e. it will either be modified or not modified, but not in
    # between, during a system crash (i.e. preemtion)
    checkpoint_path = os.path.join(checkpoint_path, name)
    os.replace(temp_path, checkpoint_path)
    msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Checkpoint saved at " + checkpoint_path
    logging.info(msg)

def load_checkpoint(cfg: omegaconf.DictConfig, model, optimizer):
    """Load checkpoint."""
    name = name_from_config(cfg)
    checkpoint_path = os.path.join(OUTPUT_DIR, 'checkpoints', name)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # model.load_checkpoint(checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        batch_step = checkpoint['batch_step']
        curr_loss = checkpoint['cur_loss']
        rng = checkpoint['rng']
        torch.random.set_rng_state(rng)
        logging.info(f"training state restored at batch_step={batch_step}")
    else:
        batch_step = 0
        curr_loss = 0.0
        logging.info(f"No checkpoint detected at {checkpoint_path} \nStarting from initial state.")
    return model, optimizer, batch_step, curr_loss