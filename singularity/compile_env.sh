#!/bin/bash
rm -rf sandbox
rm -rf pycilt.sif
singularity build -s ./sandbox ./pycilt.def
singularity build ./pycilt.sif ./sandbox/
rm -rf sandbox