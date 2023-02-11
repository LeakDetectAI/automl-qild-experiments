#!/bin/bash

rsync -avz -P --rsh="ssh -o StrictHostKeyChecking=no -l prithag" --exclude=".git" --exclude="results" --exclude="*.log" --exclude="*.pkl" --exclude="*.h5" --exclude="build"  --exclude="dist" --exclude="pycild.egg-info" --exclude=".~lock." --exclude=".idea" --exclude="*.pyc" --exclude="*.gitignore" \
--exclude="\*\sandbox" n2login1.ab2021.pc2.uni-paderborn.de:/scratch/hpc-prf-autosca/prithag/information-leakage-techniques/experiments/mutual_information/logs ~/git/information-leakage-techniques/experiments/mutual_information/ 