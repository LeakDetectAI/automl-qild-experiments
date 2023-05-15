#!/bin/bash

rsync -avz -P --rsh="ssh -o StrictHostKeyChecking=no -l prithag" --exclude=".git" --exclude="*.log" --exclude="*.pkl" --exclude="*.h5" \
--exclude="build"  --exclude="dist" --exclude="pycild.egg-info" --exclude=".~lock." --exclude=".idea" --exclude="*.pyc" --exclude="*.gitignore" \
--exclude="/experiments/automl" --exclude="wandb/" ~/git/information-leakage-techniques n2login1.ab2021.pc2.uni-paderborn.de:/scratch/hpc-prf-autosca/prithag/