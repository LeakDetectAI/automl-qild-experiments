#!/bin/bash

for i in {0..19}; do
  sbatch scripts/create_final_results.sh "leakage_detection_padding" $i
done
#for i in {0..19}; do
#  sbatch scripts/create_final_results.sh "leakage_detection" $i
#done
#for i in {0..19}; do
#  sbatch scripts/create_final_results.sh "leakage_detection_new" $i
#done
