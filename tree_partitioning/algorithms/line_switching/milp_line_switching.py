#!/usr/bin/env ipython
from tree_partitioning.classes import Case, TreePartition, Partition, SwitchedLines


def milp_ls(partition: Partition):
    """
    Solves the Line Switching Problem using MILP
    and returns the corresponding Tree Partition.
    """

    switched_lines = SwitchedLines(...)
    return TreePartition(partition, switched_lines)
