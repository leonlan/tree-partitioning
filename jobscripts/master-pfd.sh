#!/usr/bin/env bash
# Master script to run experiments for tree partitioning power flow disurption
instances=( $(ls -1 ./instances/*) )

echo $instances
for inst in $instances
do
	bn=$(basename ${inst})
	sbatch --job-name=$bn --output=$bn.out --export=inst=$instance jobscripts/sub-pfd.sh
	echo $inst
done
