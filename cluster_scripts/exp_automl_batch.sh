#!/bin/bash

count=$1
for i in $(seq $count);do
    sbatch --begin=now+3hours scripts/experiment_run_gpu.sh 'automl'
    sbatch --begin=now+3hours scripts/experiment_run.sh 'automl'
done