#!/bin/bash

count=$1
for i in $(seq $count);do
    #sbatch scripts/experiment_run_gpu.sh 'automl'
    sbatch scripts/experiment_run_gpu2.sh 'automl'
    sbatch scripts/experiment_run.sh 'automl'
done