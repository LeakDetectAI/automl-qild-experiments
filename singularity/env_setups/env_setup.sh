#!/bin/bash

ml load system singularity
export IMG_FILE=$PFS_FOLDERA/information-leakage-techniques/singularity/pycilt.sif
singularity exec -B $PFS_FOLDERA/information-leakage-techniques/ --nv $IMG_FILE poetry run pip install pyreadline
singularity exec -B $PFS_FOLDERA/information-leakage-techniques/ --nv $IMG_FILE poetry install -vvv
