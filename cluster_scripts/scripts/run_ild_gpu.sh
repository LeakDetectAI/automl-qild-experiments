#!/bin/bash
#SBATCH -J "ILD"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -A hpc-prf-autosca
#SBATCH -t 4-00:00:00
#SBATCH -p gpu
#SBATCH --mail-user prithag@mail.uni-paderborn.de
#SBATCH -o /scratch/hpc-prf-autosca/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-autosca/prithag/clusterout/%x-%j


cd $PFS_FOLDER/information-leakage-techniques/
ml lang
ml Python/3.9.5
ml Python/3.9.5-GCCcore-10.3.0

export PYTHONUSERBASE=$PFS_FOLDER/.local
export PATH=$PFS_FOLDER/.bin:$PATH
export PATH=$PFS_FOLDER/.local/bin:$PATH
which python
which pip

export SCRIPT_FILE=$PFS_FOLDER/information-leakage-techniques/cluster_script_ild.py
python $SCRIPT_FILE --cindex=$SLURM_JOB_ID --isgpu=1 --schema='leakage_detection'

exit 0
~