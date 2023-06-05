count=$1
for i in $(seq $count);do
    sbatch scripts/run_ild.sh 'leakage_detection'
    sbatch scripts/run_ild_largemem.sh 'leakage_detection'
    sbatch scripts/run_ild_gpu.sh 'leakage_detection'

done
