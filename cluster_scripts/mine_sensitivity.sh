#!/bin/sh
#SBATCH -J "MineSensitivity"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -A hpc-prf-autosca
#SBATCH -t 1-00:00:00
#SBATCH -q express
#SBATCH -p normal
#SBATCH -o /scratch/hpc-prf-autosca/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-autosca/prithag/clusterout/%x-%j

cd $PFS_FOLDER/information-leakage-techniques/
module reset
ml lang
ml Python/3.9.5
source ~/.bashrc
which python
which pip

export SCRIPT_FILE=$PFS_FOLDER/information-leakage-techniques/mine_sensitivity_analysis.py
python $SCRIPT_FILE

exit 0
~