from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandapower as pp

from .Case import Case
from .Partition import Partition
from .ReducedGraph import ReducedGraph


class Solution:
    """
    A Solution is represented by a tree partition.
    """

    def __init__(
        self,
        case: Case,
        generators,
        partition: Partition,
        switched_lines: list,
    ):
        self.G = case.G
        self.net = case.net
        self.partition = partition
        self.switched_lines = switched_lines
        self.generators = generators

        self.post_switching_graph = self.G.copy()
        self.post_switching_graph.remove_edges_from(switched_lines)

    def plot(self, path: str, show=False, post=True):
        nc = [
            "#ea9999",  # red
            "#a4c2f4",
            "#b6d7a8",
            "#b4a7d6",  # purple
            "#f9cb9c",
            "#eeeeee",  # Grey
        ]

        ec = ["#db4c39", "#2a7afa", "#8fd46a", "#595959"]

        # # Undirected graph has better visuals
        G = self.G.to_undirected()

        # TODO for all lines  that there is only k-1 with 2 colors.

        # Post-bbd partition colors
        vertex_colors = [nc[self.partition.membership[v]] for v in G.nodes]
        fig, ax = plt.subplots(figsize=[16, 12])
        pos = nx.kamada_kawai_layout(self.post_switching_graph, weight=None)

        # Draw buses
        factor = min(1, max(0.3, 200 / len(G)))
        buses = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=250 * factor,
            node_color=vertex_colors,
            linewidths=2,
        )
        # nx.draw_networkx_labels(
        #     G,
        #     pos,
        #     {i: str(i) for i in G.nodes if i not in self.generators.keys()},
        #     font_size=9,
        # )
        nx.draw_networkx_labels(
            G,
            pos,
            {node: "G" for group in self.generators.values() for node in group},
            # {i: "G" for i in self.generators.keys()},
            font_size=9,
        )

        # Draw regular lines
        lines = nx.draw_networkx_edges(
            G,
            pos,
            # edgelist=[e for e in G.edges if e not in ],
            edgelist=[e for e in G.edges if e not in self.switched_lines],
            width=2.5,
            edge_color="#595959",
        )

        # Draw switched lines
        switched_lines = nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[e for e in G.edges if e in self.switched_lines],
            width=2,
            alpha=0.8,
            edge_color="red",
            style="dashed",
        )

        # Gives bus stroke colors
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#595959")
        ax.set_axis_off()

        if show:
            plt.show()

        plt.savefig(path)
