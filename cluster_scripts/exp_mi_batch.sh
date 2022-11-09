count=4
for i in $(seq $count);do
    sbatch mi_experiment_run.sh
done