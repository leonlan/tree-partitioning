from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandapower as pp

from .Case import Case
from .ReducedGraph import ReducedGraph


class Solution:
    """
    A Solution is represented by a tree partition.
    """

    def __init__(
        self,
        partition: Partition,
        switched_lines: list,
        results: dict | None = None,
        model=None,
    ):
        case = Case()
        self.G = case.G
        self.net = case.net
        self.partition = partition
        self.switched_lines = switched_lines

        self.post_switching_graph = self.G.copy()
        self.post_switching_graph.remove_edges_from(switched_lines)
        self.post_switching_net = _deactivate_lines_pp(self.net, switched_lines)

        if results is not None:
            self._results = self.set_results(results)

        if model:
            self.model = model

    def is_tree_partition(self):
        """
        Verifies that (1) the post-switching graph is connected and
        (2) P is a tree partition of the post-switching graph.
        """
        if not nx.is_weakly_connected(self.post_switching_graph):
            print("Post_switching_graph not (weakly) connected.")
            return False

        if not ReducedGraph(self.post_switching_graph, self.partition).is_tree():
            print("Post switching reduced graph not a tree.")
            return False

        return True

    @property
    def objective(self) -> float:
        if not self._objective:
            self.compute_objective()

        return self._objective

    def compute_objective(self):
        """Compute objective if not available yet."""
        pp.rundcpp(self.post_switching_net)
        self._objective = _max_loading_percent(self.post_switching_net)

    @property
    def results(self):
        """
        Calculate solution results about partitioning and line switching actions.
        """

        # Partition
        partitioning = {}
        # modularity score
        # runtime
        # n_clusters
        # # cross edges
        # Fraction of cross edges
        # # lines to be switched off
        # clusters
        # bridge-blocks
        #
        #

        # Line switching
        # #runtime
        # #spanning trees
        # objective
        # # congested lines
        # # lines to be swithced off
        # % lines switched off
        # power flow disruption
        #
        return results

    def plot(self, path: str, show=False, post=False):
        nc = [
            "#ea9999",  # red
            "#a4c2f4",
            "#b6d7a8",
            "#b4a7d6",  # purple
            "#f9cb9c",
            "#eeeeee",  # Grey
        ]

        ec = ["#db4c39", "#2a7afa", "#8fd46a", "#595959"]

        # Undirected graph has better visuals
        G = self.G.to_undirected()

        # Post-bbd partition colors
        vertex_colors = [nc[self.partition.membership[v]] for v in G.nodes]
        fig, ax = plt.subplots(figsize=[16, 12])
        pos = nx.kamada_kawai_layout(
            (self.post_switching_graph if post else self.G), weight=None
        )

        # Draw buses
        factor = min(1, max(0.3, 200 / len(G)))
        buses = nx.draw_networkx_nodes(
            G, pos, node_size=250 * factor, node_color=vertex_colors, linewidths=2,
        )
        nx.draw_networkx_labels(
            G, pos, {i: str(i) for i in G.nodes}, font_size=9,
        )

        switched_lines_idx = [line_idx for _, _, line_idx in self.switched_lines]
        # Draw regular lines
        lines = nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[
                e
                for *e, data in G.edges(data=True)
                if data["edge_index"] not in switched_lines_idx
            ],
            width=2.5,
            edge_color="#595959",
        )

        # Draw switched lines
        switched_lines = nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[
                e
                for *e, data in G.edges(data=True)
                if data["edge_index"] in switched_lines_idx
            ],
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


# FIXME: NOT DRY: also used in brute_force
def _deactivate_lines_pp(net, lines):
    """Deactivate lines of a pandapower network. """
    # Get the line names first from the lines
    netdict = Case().netdict
    line_names = [netdict["lines"][line]["name"] for line in lines]

    net = pp.copy.deepcopy(net)
    net.line.loc[net.line["name"].isin(line_names), "in_service"] = False
    net.trafo.loc[net.trafo["name"].isin(line_names), "in_service"] = False

    pp.rundcpp(net)

    return net


def _max_loading_percent(net):
    """Compute the maximum loading percent of net."""
    gl = max(net.res_line.loading_percent) / 100

    # Some cases do not have transformers
    try:
        gt = max(net.res_trafo.loading_percent) / 100
    except ValueError:
        return gl
    return max(gl, gt)
