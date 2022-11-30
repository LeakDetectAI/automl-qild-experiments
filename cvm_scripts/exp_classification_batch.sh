count=$1
for i in $(seq $count);do
    bash scripts/classification_experiment_run.sh
done