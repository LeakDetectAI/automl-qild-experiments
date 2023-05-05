#!/bin/sh
#SBATCH -J "InsertJobs"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -A hpc-prf-autosca
#SBATCH -t 10:00:00
#SBATCH -p normal
#SBATCH -o /scratch/hpc-prf-autosca/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-autosca/prithag/clusterout/%x-%j

cd $PFS_FOLDERA/information-leakage-techniques/

module reset
ml lang
ml Anaconda3
source ~/.bashrc
conda activate ild
which python
which conda
export SCRIPT_FILE=$PFS_FOLDERA/information-leakage-techniques/insert_jobs.py
python $SCRIPT_FILE

exit 0
~