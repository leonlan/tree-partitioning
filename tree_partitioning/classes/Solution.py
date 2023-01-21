from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from .Case import Case
from .Partition import Partition


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

    def plot(self, path: str, show=False, undirected=True, post=True):
        nc = [
            "#ea9999",  # red
            "#a4c2f4",  # blue
            "#b6d7a8",
            "#b4a7d6",  # purple
            "#f9cb9c",
            "#eeeeee",  # Grey
        ]

        # Undirected graph has better visuals
        G = self.G.to_undirected() if undirected else self.G

        # Post-bbd partition colors
        vertex_colors = [nc[self.partition.membership[v]] for v in G.nodes]
        _, ax = plt.subplots(figsize=[16, 12])
        pos = nx.kamada_kawai_layout(
            (self.post_switching_graph if post else G), weight=None
        )

        # Draw buses
        factor = min(1, max(0.3, 200 / len(G)))
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=250 * factor,
            node_color=vertex_colors,
            linewidths=2,
        )

        # Draw generator labels
        nx.draw_networkx_labels(
            G,
            pos,
            {node: "G" for group in self.generators.values() for node in group},
            font_size=9,
        )

        # Draw regular lines
        nx.draw_networkx_edges(
            G,
            pos,
            # edgelist=[e for e in G.edges if e not in ],
            edgelist=[e for e in self.G.edges if e not in self.switched_lines],
            width=2.5,
            edge_color="#595959",
        )

        # Draw switched lines
        nx.draw_networkx_edges(
            G,
            pos,
            # NOTE We use self.G.edges here because they are directed, just like self.switched_lines
            edgelist=[e for e in self.G.edges if e in self.switched_lines],
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
