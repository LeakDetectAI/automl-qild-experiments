#!/bin/bash

count=$1
count2=$2
count3=$3
for i in $(seq $count);do
    sbatch --begin=2024-04-13T08:00:00 scripts/run_ild.sh
    #sbatch scripts/run_ild_largemem.sh
done
for i in $(seq $count2);do
    sbatch --begin=2024-04-13T08:00:00 scripts/run_ild_n_gpu.sh
    #sbatch scripts/run_ild_l_gpu.sh
done

for i in $(seq $count3);do
    sbatch --begin=2024-04-13T08:00:00 scripts/run_ild_gpu.sh
done