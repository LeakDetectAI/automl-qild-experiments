count=$1
for i in $(seq $count);do
    bash scripts/mi_experiment_run.sh
done