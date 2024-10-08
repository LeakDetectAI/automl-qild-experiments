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
cd $PFS_FOLDER/automl_qild_experiments/
ml lang
ml Python/3.9.5
ml Python/3.9.5-GCCcore-10.3.0

export PYTHONUSERBASE=$PFS_FOLDER/automl_qild_experiments/.local
export PATH=$PFS_FOLDER/automl_qild_experiments/.bin:$PATH
export PATH=$PFS_FOLDER/automl_qild_experiments/.local/bin:$PATH
which python
which pip

export SCRIPT_FILE=$PFS_FOLDER/automl_qild_experiments/cluster_script.py
python $SCRIPT_FILE --cindex=$SLURM_JOB_ID --isgpu=0 --schema=$1

exit 0
~