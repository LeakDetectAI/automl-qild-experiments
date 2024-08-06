#!/bin/sh
#SBATCH -J "MineSensitivity"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -A hpc-prf-aiafs
#SBATCH -t 1-00:00:00
#SBATCH -q express
#SBATCH -p normal
#SBATCH -o /scratch/hpc-prf-aiafs/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-aiafs/prithag/clusterout/%x-%j

cd $PFS_FOLDER/automl_qild_experiments/
module reset
ml lang
ml Python/3.9.5
source ~/.bashrc
which python
which pip

export SCRIPT_FILE=$PFS_FOLDER/automl_qild_experiments/mine_sensitivity_analysis.py
python $SCRIPT_FILE

exit 0
~