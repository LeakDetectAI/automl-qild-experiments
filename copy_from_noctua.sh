#!/bin/bash

#rsync -avz -P --rsh="ssh -o StrictHostKeyChecking=no -l prithag" --exclude=".git" --exclude="build"  --exclude="dist" --exclude="pycild.egg-info" --exclude=".~lock." --exclude=".idea" --exclude="*.pyc" --exclude="*.gitignore" 
#--exclude="\*\sandbox" n2login1.ab2021.pc2.uni-paderborn.de:/scratch/hpc-prf-aiafs/prithag/automl_qild_experiments/experiments/automl/logs ~/git/automl_qild_experiments/experiments/automl/logs

rsync -avz --rsh="ssh -o StrictHostKeyChecking=no -l prithag" --include '*/' --include '*.txt' --exclude '*' n2login1.ab2021.pc2.uni-paderborn.de:/scratch/hpc-prf-aiafs/prithag/automl_qild_experiments/ ~/git/automl_qild_experiments/
rsync -avz --rsh="ssh -o StrictHostKeyChecking=no -l prithag" --include '*/' --include '*.txt' --exclude '*' n2login1.ab2021.pc2.uni-paderborn.de:/scratch/hpc-prf-aiafs/prithag/automl-qild/ ~/git/automl-qild/
