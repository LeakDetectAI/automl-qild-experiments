#!/bin/bash

rsync -avz -P --rsh="ssh -o StrictHostKeyChecking=no -l prithag" --exclude=".git" --exclude="build"  --exclude=".ipynb_checkpoints" --exclude="pycild.egg-info" \
--exclude="*.log" --exclude=".~lock." --exclude=".idea" --exclude="*.pyc" --exclude="*.gitignore" --exclude="experiments/leakage_detection/" \
--exclude="wandb/" ~/git/automl_quant_il_detect n2login1.ab2021.pc2.uni-paderborn.de:/scratch/hpc-prf-aiafs/prithag/