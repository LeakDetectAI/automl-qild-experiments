count=10
for i in $(seq $count);
do
    sbatch classification_experiment_run.sh
done