#!/bin/bash

ml lang
ml Python/3.9.5
ml Python/3.9.5-GCCcore-10.3.0

export PYTHONUSERBASE=$PFS_FOLDER/.local
export PATH=$PFS_FOLDER/.bin:$PATH
export PATH=$PFS_FOLDER/.local/bin:$PATH
which python
which pip
pip install -e ./