#!/bin/bash
#SBATCH -J "MIEstimation"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -A hpc-prf-autosca
#SBATCH -t 2-00:00:00
#SBATCH -p normal
#SBATCH -o /scratch/hpc-prf-autosca/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-autosca/prithag/clusterout/%x-%j

cd $PFS_FOLDER/information-leakage-techniques/
module reset
export SCRIPT_FILE=$PFS_FOLDER/information-leakage-techniques/cluster_script.py
ml lang
ml Anaconda3
source ~/.bashrc
conda activate ild
which python
which conda
python $SCRIPT_FILE --cindex=$SLURM_JOB_ID --isgpu=0 --schema=$1

exit 0
~