
for i in vm vm1 vm2 vm3 vm4
do
	echo "field= " $i
	rsync -avz -P --rsh="sshpass -p inspiron ssh -o StrictHostKeyChecking=no -l prithagupta" --exclude=".git" $i:~/automl_quant_il_detect/experiments/mutual_information/ ~/git/automl_quant_il_detect/experiments/mutual_information/
	echo "******************************************************************************************************************"
done


