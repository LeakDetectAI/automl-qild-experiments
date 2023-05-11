count=$1
for i in $(seq $count);do
    sbatch scripts/experiment_run.sh 'mutual_information'
done