#!/bin/bash
poetry run python -m tree_partitioning.transient_stability --time_limit 1200 --min_size $1 --max_size $2 --n_clusters $3 --results_path "$4"
