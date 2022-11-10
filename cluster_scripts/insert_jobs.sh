#!/bin/sh
#SBATCH -J "InsertJobs"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -A hpc-prf-obal
#SBATCH -t 2-00:00:00
#SBATCH -p largemem
#SBATCH -o /scratch/hpc-prf-obal/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-obal/prithag/clusterout/%x-%j

cd $PFS_FOLDER/information-leakage-techniques/
module reset
module load system singularity
export IMG_FILE=$PFS_FOLDER/information-leakage-techniques/singularity/pycilt.sif
export SCRIPT_FILE=$PFS_FOLDER/information-leakage-techniques/insert_jobs.py


module list
singularity exec -B $PFS_FOLDER/information-leakage-techniques/ --nv $IMG_FILE pipenv run python $SCRIPT_FILE

exit 0
~