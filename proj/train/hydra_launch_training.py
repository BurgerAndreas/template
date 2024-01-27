import os
import hydra
import multiprocessing
from omegaconf import DictConfig, OmegaConf
import wandb

from proj.utils.checkpointing import name_from_config
from proj.train.train import train_loop

cwd = os.getcwd()


@hydra.main(config_name="palix", config_path="config", version_base="1.3")
def run_train(cfg: dict) -> None:
    """Run training loop.
    Usage:
        python skills/hydra_launch_train.py
        python skills/hydra_launch_train.py +experiment=debugging
        python skills/hydra_launch_train.py wandb=True model.batch_size=64
    """

    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    # print(OmegaConf.to_yaml(cfg))

    if cfg.wandb:
        wandb.login()
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        run = wandb.init(name=name_from_config(cfg=cfg), project="skills")
    else:
        wandb.init(mode="disabled")
        os.environ["WANDB_DISABLED"] = "true"
    
    # set env flags
    if cfg.debugging == True:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '1'

    cfg["cwd"] = cwd
    with open("job_config.json", "w") as fp:
        OmegaConf.save(config=cfg, f=fp.name)

    train_loop(cfg)

    return


if __name__ == "__main__":
    run_train()