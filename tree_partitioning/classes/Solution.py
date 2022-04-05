from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from .Case import Case
from .ReducedGraph import ReducedGraph


class Solution:
    """
    A Solution is represented by a tree partition.
    """

    def __init__(self, partition: Partition, switched_lines: SwitchedLines):
        self.G = Case().G
        self.partition = partition
        self.switched_lines = switched_lines
        self.post_switching_graph = self.G.copy()
        self.post_switching_graph.remove_edges_from(switched_lines.lines)

    def is_tree_partition(self):
        """
        Verifies that (1) the post-switching graph is connected and
        (2) P is a tree partition of the post-switching graph.
        """
        return (
            nx.is_weakly_connected(self.post_switching_graph)
            and ReducedGraph(self.post_switching_graph, self.partition).is_tree()
        )

    def plot(self, path: str, show=False):
        nc = [
            "#ea9999",  # red
            "#a4c2f4",
            "#b6d7a8",
            "#b4a7d6",  # purple
            "#f9cb9c",
            "#eeeeee",  # Grey
        ]

        ec = ["#db4c39", "#2a7afa", "#8fd46a", "#595959"]

        # Post-bbd partition colors
        vertex_colors = [nc[self.partition.membership[v]] for v in self.G.nodes]
        fig, ax = plt.subplots(figsize=[16, 12])
        pos = nx.kamada_kawai_layout(self.G, weight=None)

        # Draw buses
        buses = nx.draw_networkx_nodes(
            self.G, pos, node_size=250, node_color=vertex_colors, linewidths=2,
        )

        # Draw regular lines
        lines = nx.draw_networkx_edges(
            self.G,
            pos,
            edgelist=[
                e for e in self.G.edges if list(e) not in self.switched_lines.lines
            ],
            arrows=False,
            width=2.5,
            edge_color="#595959",
        )

        # Draw switched lines
        switched_lines = nx.draw_networkx_edges(
            self.G,
            pos,
            edgelist=[e for e in self.G.edges if list(e) in self.switched_lines.lines],
            arrows=False,
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
