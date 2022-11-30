#!/bin/sh
#SBATCH -J "MutualInformationGPU"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH -A hpc-prf-autosca
#SBATCH -t 10:00:00
#SBATCH -p gpu
#SBATCH -o /scratch/hpc-prf-autosca/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-autosca/prithag/clusterout/%x-%j

cd $PFS_FOLDERA/information-leakage-techniques/
module reset
module load system singularity
#module --ignore_cache load "system/singularity/3.10.2.lua"
export IMG_FILE=$PFS_FOLDERA/information-leakage-techniques/singularity/pycilt.sif
export SCRIPT_FILE=$PFS_FOLDERA/information-leakage-techniques/cluster_script.py


module list
singularity exec -B $PFS_FOLDERA/information-leakage-techniques/ --nv $IMG_FILE pipenv run python $SCRIPT_FILE --cindex=$SLURM_JOB_ID --isgpu=1 --schema="mutual_information"

exit 0
~