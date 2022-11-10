#!/bin/bash

ml load system singularity
rm -rf .venv/
export IMG_FILE=$PFS_FOLDER/information-leakage-techniques/singularity/pycilt.sif
singularity exec -B $PFS_FOLDER/information-leakage-techniques/ --nv $IMG_FILE pipenv install