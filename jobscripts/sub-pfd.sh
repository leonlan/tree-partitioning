#!/usr/bin/env/ bash
#SBATCH --partition=normal
#SBATCH --constraint=silver_4110
#SBATCH --nodes 1
#SBATCH -t 2:00:00
algs=(
    "single_stage"
    "two_stage"
    "recursive"
    "warm_start"
)

for k in {2..5}
do
    for alg in "${algs[@]}"; do
        poetry run python experiments/power_flow_disruption.py --min_clusters k --max_clusters k --algorithm $alg --instance_pattern $inst
done
done

