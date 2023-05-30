count=$1
for i in $(seq $count);do
    #sbatch scripts/automl_experiment_run.sh
    sbatch scripts/experiment_run_ild.sh 'leakage_detection'
done