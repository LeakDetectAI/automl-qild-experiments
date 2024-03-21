#!/bin/bash

count=$1
count2=$2
for i in $(seq $count);do
    sbatch scripts/run_ild.sh
    sbatch scripts/run_ild_largemem.sh
done
for i in $(seq $count2);do
    sbatch scripts/run_ild_gpu.sh
    sbatch scripts/run_ild_n_gpu.sh
    sbatch scripts/run_ild_l_gpu.sh
done