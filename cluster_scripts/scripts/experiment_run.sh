#!/bin/bash
#SBATCH -J "AutoML"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -A hpc-prf-aiafs
#SBATCH -t 3-00:00:00
#SBATCH -p normal
#SBATCH -o /scratch/hpc-prf-aiafs/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-aiafs/prithag/clusterout/%x-%j


#largemem
cd $PFS_FOLDER/automl_quant_il_detect/
ml lang
ml Python/3.9.5
echo $PYTHONUSERBASE
echo $PATH
which python
which pip

export SCRIPT_FILE=$PFS_FOLDER/automl_quant_il_detect/cluster_script.py
python $SCRIPT_FILE --cindex=$SLURM_JOB_ID --isgpu=0 --schema=$1

exit 0
~