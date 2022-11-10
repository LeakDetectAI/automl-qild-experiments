count=$1
for i in $(seq $count);do
    sbatch scripts/mi_experiment_gpu_run.sh
done