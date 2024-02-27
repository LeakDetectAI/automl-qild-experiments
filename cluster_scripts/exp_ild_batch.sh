#!/bin/bash

count=$1
for i in $(seq $count);do
    sbatch scripts/run_ild.sh
    #sbatch scripts/run_ild_largemem.sh
    sbatch scripts/run_ild_gpu.sh
done
