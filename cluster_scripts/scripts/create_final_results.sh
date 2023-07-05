#!/bin/sh
#SBATCH -J "CreateResults"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -A hpc-prf-autosca
#SBATCH -t 1-00:00:00
#SBATCH -p normal
#SBATCH -o /scratch/hpc-prf-autosca/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-autosca/prithag/clusterout/%x-%j

cd $PFS_FOLDER/information-leakage-techniques/
ml lang
ml Python/3.9.5
echo $PYTHONUSERBASE
echo $PATH
which python
which pip

export SCRIPT_FILE=$PFS_FOLDER/information-leakage-techniques/create_final_results.py
python $SCRIPT_FILE --schema=$1 --bucket_id=$2

exit 0
~