# Template for hydra and wandb

## Installation
```bash
python3.10 -m venv venv
source venv/bin/activate 

pip install --upgrade pip

# check cuda version first with nvidia-smi
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tensorflow[and-cuda] --index-url https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-2.15.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

pip install hydra-core hydra-colorlog wandb omegaconf

pip install -e .
```

## Usage
```bash
python skills/hydra_launch_training.py
python skills/hydra_launch_training.py +experiment=debugging
python skills/hydra_launch_training.py wandb=True model.batch_size=64
```

## Slurm usage
```bash
sbatch launch_job.slrm
```