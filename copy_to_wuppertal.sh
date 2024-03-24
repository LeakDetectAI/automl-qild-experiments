#!/bin/bash

rsync -avz -P --rsh="ssh -o StrictHostKeyChecking=no -l pritha" --exclude=".git" --exclude="build"  --exclude=".ipynb_checkpoints" --exclude="pycild.egg-info" \
--exclude=".~lock." --exclude=".idea" --exclude="*.pyc" --exclude="*.gitignore" --exclude="experiments/leakage_detection/" \
--exclude="wandb/" ~/git/automl_quant_il_detect sca:/home/pritha/