#!/bin/bash

ml load system singularity
rm -rf .venv/
export IMG_FILE=$PFS_FOLDERA/information-leakage-techniques/singularity/pycilt_d.sif
singularity exec -B $PFS_FOLDERA/information-leakage-techniques/ --nv $IMG_FILE pipenv install