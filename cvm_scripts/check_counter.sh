#!/bin/bash
val=`cat ~/.counter`
export CCS_REQID=$HOST$val
val=$(($val+1))
echo $val > ~/.counter
echo "Checking if we updated the counter in file"
cat ~/.counter
echo "CCS_REQID = " $CCS_REQID
echo 1 > ~/.counter
echo "Reset"
cat ~/.counter

