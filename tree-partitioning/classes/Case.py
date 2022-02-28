#!/usr/bin/env ipython
"""
Grid class that contains all representations of the power grid, including:
- pandapower network
- igraph
- line dataframe snapshot
- bus dataframe snapshot
"""
from collections import namedtuple
import itertools
import os
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.converter as pc
import pandapower.networks as pn

from sklearn.cluster import SpectralClustering
import scipy
import scipy.sparse
from scipy.sparse import csr_matrix as csr
import time

from .nx_utils import a
from .pp_utils import _load_pp_case


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            up = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = up

        return cls._instances[cls]

    def clear(cls):
        try:
            del Singleton._instances[cls]
        except KeyError:
            pass


class Case(metaclass=Singleton):
    """
    Test case of a single power network.

    This class contains multiple representation of the network, among others:
    - a pandapower power network
    - a networkx graph
    - a pandas df representation

    These multiple representations are helpful to deal with various aspects of
    the tree partitioning implementation, such as graph partitioning and
    power network simulations.
    """

    def __init__(self, path):
        self.net = _load_pp_case(path)
        self.G = ...
        self.df = ...

    def deactivate_line(self, line):
        ...


class Case:
    """Case object containg all power grids representations.

    - net: pandapower network
    - df: pandas DataFrame of all the lines
    - igg: igraph
    - G: networkx graph

    """

    def __init__(
        self,
        case,
        merge_lines=False,
        weight="b",
        opf_flag=False,
        ac=False,
        pn_case=False,
        init_compute_matrices=False,
        susceptance_method="pandapower",
    ):
        """
        - merge_lines (bool): Return the network representation with merged parallel lines
        - init_compute_matrices (bool): Indicate if the matrices should be computed at initialization
        """
        self.path, self.name, self.pn = case
        self.net, self.df, self.df_bus, self.igg, self.G = load_case(
            case,
            opf_flag=opf_flag,
            ac=ac,
            pn_case=pn_case,
            susceptance_method=susceptance_method,
        )

        if merge_lines:
            _, self.merged_df, self.df_bus, self.merged_igg, self.merged_G = load_case(
                case,
                merge_lines=merge_lines,
                opf_flag=opf_flag,
                ac=ac,
                pn_case=pn_case,
                susceptance_method=susceptance_method,
            )

        if init_compute_matrices:
            self.compute_matrices(weight)

    def __name__(self):
        return f"Case object for test case {self.name}."

    def __repr__(self):
        return f"Case object for test case {self.name}."

    def compute_matrices(self, weight="b"):
        """Computes the matrices associated with the network."""
        self.C = nx.linalg.graphmatrix.incidence_matrix(self.merged_G, oriented=True)
        self.B = np.diag([x[-1] for x in self.merged_G.edges.data(weight, default=1)])
        self.L = scipy.sparse.csr_matrix.toarray(
            nx.laplacian_matrix(self.merged_G, weight=weight)
        )
        self.Linv = np.linalg.pinv(self.L)
        # D := PTDF matrix
        self.D = self.B @ self.C.T @ self.Linv @ self.C
        self.DD = np.diag(self.D) * np.ones(self.D.shape)

        # K := LODF matrix
        # - Diagonal entries are 1
        # - Bridges are filled with NaN
        self.K = np.divide(
            self.D,
            (1 - self.DD),
            out=np.nan * np.ones(self.D.shape),
            where=~np.isclose(self.DD, 1),
        )
        # Permutate the matrix

        np.fill_diagonal(self.K, 1)
        O = np.ones(self.Linv.shape)
        self.Linvaa = np.diag(self.Linv).reshape((len(self.Linv), 1)) * O  # L+aa matrix
        self.Linvbb = np.diag(self.Linv) * O  # L+bb matrix
        self.R = self.Linvaa + self.Linvbb - 2 * self.Linv

    def deactivate_lines(self, L):
        """Deactivate a set of lines.

        Modifies all representations of the network.

        # TODO: Currently does not work yet for the merged lines objects.

        L (list): List of line names
        """
        link_idx = self.df[self.df["name"].isin(L)].edge_index
        link_nx_ids = self.df[self.df["name"].isin(L)].nx_id

        # Deactivate links in pp.net
        self.net.line.loc[self.net.line["name"].isin(L), "in_service"] = False
        self.net.trafo.loc[self.net.trafo["name"].isin(L), "in_service"] = False

        # Deactivate links in df
        self.df.loc[self.df["name"].isin(L), "in_service"] = False

        # Deactivate edges in networkx
        for (i, j, k) in link_nx_ids:
            try:
                self.G.remove_edge(i, j, k)
            except nx.NetworkXError:
                raise "The given edges have already been removed."

        # Deactivate edges in igraph
        self.igg.delete_edges(link_idx)


## Previously in constants.py

TEST_CASE = namedtuple("TEST_CASE", "path, name, pn")
DATA_DIR = Path("data/")


def create(file_name, name, pn=None):
    """Creates a Testcase namedtuple for the case.

    Calculates the full path and gives upper cased name.
    """
    path = DATA_DIR / Path(f"pglib_opf_case{file_name}.mat")
    return TEST_CASE(path, name, pn)
