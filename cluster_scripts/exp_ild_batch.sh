#!/bin/bash

count=$1
# shellcheck disable=SC2034

for i in $(seq $count);do
    sbatch scripts/run_ild.sh
    #sbatch scripts/run_ild_largemem.sh
done

#count2=$2
#for i in $(seq $count2);do
#    sbatch scripts/run_ild_n_gpu.sh
#    sbatch scripts/run_ild_l_gpu.sh
#done

#count3=$3
#for i in $(seq $count3);do
#    sbatch scripts/run_ild_gpu.sh
#    pass
#done