#!/bin/sh
#SBATCH -J "InsertJobs"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -A hpc-prf-aiafs
#SBATCH -t 10:00:00
#SBATCH -p normal
#SBATCH -o /scratch/hpc-prf-aiafs/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-aiafs/prithag/clusterout/%x-%j

cd $PFS_FOLDER/automl_quant_il_detect/
ml lang
ml Python/3.9.5
ml Python/3.9.5-GCCcore-10.3.0

export PYTHONUSERBASE=$PFS_FOLDER/automl_quant_il_detect/.local
export PATH=$PFS_FOLDER/automl_quant_il_detect/.bin:$PATH
export PATH=$PFS_FOLDER/automl_quant_il_detect/.local/bin:$PATH
which python
which pip

export SCRIPT_FILE=$PFS_FOLDER/automl_quant_il_detect/insert_jobs.py
python $SCRIPT_FILE

exit 0
~