count=$1
for i in $(seq $count);do
    sbatch scripts/experiment_run_mi.sh 'mutual_information'
done