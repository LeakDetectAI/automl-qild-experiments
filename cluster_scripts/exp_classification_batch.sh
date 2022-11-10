count=$1
for i in $(seq $count);do
    sbatch scripts/classification_experiment_run.sh
done