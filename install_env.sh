#!/bin/bash

# Load required modules
ml lang
ml Python/3.9.5
ml Python/3.9.5-GCCcore-10.3.0

# Check and create $PFS_FOLDER/.local if it doesn't exist
if [ ! -d "$PFS_FOLDER/automl_quant_il_detect/.local" ]; then
    mkdir -p "$PFS_FOLDER/automl_quant_il_detect/.local"
fi

# Check and create $PFS_FOLDER/.bin if it doesn't exist
if [ ! -d "$PFS_FOLDER/automl_quant_il_detect/.bin" ]; then
    mkdir -p "$PFS_FOLDER/automl_quant_il_detect/.bin"
fi

# Check and create $PFS_FOLDER/.bin if it doesn't exist
if [ ! -d "$PFS_FOLDER/.cache" ]; then
    mkdir -p "$PFS_FOLDER/.cache"
fi
# Check and create $PFS_FOLDER/.bin if it doesn't exist
if [ ! -d "$PFS_FOLDER/tmp" ]; then
    mkdir -p "$PFS_FOLDER/tmp"
fi
# Set environment variables
export PYTHONUSERBASE=$PFS_FOLDER/automl_quant_il_detect/.local
export PATH=$PFS_FOLDER/automl_quant_il_detect/.bin:$PATH
export PATH=$PFS_FOLDER/automl_quant_il_detect/.local/bin:$PATH

# Display the paths of the Python and pip executables
which python
which pip

# Install the current directory as a package
pip install -e ./
