#!/bin/bash

rsync -avz -P --rsh="ssh -o StrictHostKeyChecking=no -l prithag" --exclude=".git" --exclude="results" --exclude="*.log" --exclude="build"  --exclude="dist" --exclude=".egg-info" --exclude=".~lock." --exclude=".idea" --exclude="*.pyc" --exclude="*.gitignore" \
--exclude="\*\sandbox" ~/deep-learning-sca n2login1.ab2021.pc2.uni-paderborn.de:/scratch/hpc-prf-obal/prithag/