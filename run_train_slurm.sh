#!/bin/sh

# This is the script to perform training, the goal is that code in
# this script can be safely preempted. Jobs in slurm queue are scheduled
# and preempted according to their priorities. Note that even the job with
# deadline queue can be preempted over time so it is crucial that you
# checkpoint your program state for every fixed interval: e.g 10 mins.

# Vector provides a fast parallel filesystem local to the GPU nodes,  dedicated
# for checkpointing. It is mounted under /checkpoint. It is strongly
# recommended that you keep your intermediary checkpoints under this directory
# i.e. /checkpoint/${USER}/${SLURM_JOB_ID}

# We also recommend users to create a symlink of the checkpoint dir so your
# training code stays the same with regards to different job IDs and it would
# be easier to navigate the checkpoint directory
# both are directories
# mkdir $PWD/output/checkpoint/slurm
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/output/checkpoint/slurm


# In the future, the checkpoint directory will be removed immediately after the
# job has finished. If you would like the file to stay longer, and create an
# empty "delay purge" file as a flag so the system will delay the removal for
# 48 hours
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE

# prepare the environment, here I am using environment modules, but you could
# select the method of your choice (but note that code in ~/.bash_profile or
# ~/.bashrc will not be executed with a new job)
# module purge && module load  tensorflow2-gpu-cuda10.1-conda-python3.6
source /h/burgeran/project/venv/bin/activate/venv

# Then we run our training code, using the checkpoint dir provided 
# the cod3 demonstrates how to perform checkpointing in pytorch, 
# please navigate to the file for more information.
python skills/train/hydra_launch_train.py 

# copy the checkpoints to the home directory
cp /checkpoint/${USER}/${SLURM_JOB_ID}/* $HOME/skills/output/checkpoint/

# Finally, we remove the delay purge file so the system can remove the
# checkpoint directory
rm /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE