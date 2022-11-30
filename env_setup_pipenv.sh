#!/bin/bash

module --ignore_cache load "system/singularity/3.10.2.lua"
rm -rf .venv/
export IMG_FILE=$PFS_FOLDERA/information-leakage-techniques/singularity/pycilt.sif
singularity exec -B $PFS_FOLDERA/information-leakage-techniques/ --nv $IMG_FILE pipenv install