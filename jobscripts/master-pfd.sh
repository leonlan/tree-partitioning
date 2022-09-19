#!/usr/bin/env bash
# Master script to run experiments for tree partitioning power flow disurption
instances=( $(ls -1 ./instances/*) )

for inst in $instances
do
    sbatch --job-name=$inst --output=$inst.out --export=inst=$instance sub-pfd.sh
done
