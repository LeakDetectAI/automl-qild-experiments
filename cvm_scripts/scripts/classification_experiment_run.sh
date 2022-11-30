#!/bin/sh

val=`cat ~/.counter`
export CCS_REQID=$HOSTNAME$val
val=$(($val+1))
echo "Current Counter = " cat ~/.counter
echo $val > ~/.counter
echo "Checking if we updated the counter in file"
cat ~/.counter
echo "CCS_REQID = " $CCS_REQID
export SCRIPT_FILE=~/information-leakage-techniques/cluster_script.py

conda init
conda activate ild
python $SCRIPT_FILE --cindex=$CCS_REQID --isgpu=0 --schema="classification"

echo $CCS_REQID
mail -s "Jobstatus $CCS_REQID" prithag@mail.upb.com <<< "Finished $CCS_REQID"
val=`cat ~/.hash_value`
echo $val
