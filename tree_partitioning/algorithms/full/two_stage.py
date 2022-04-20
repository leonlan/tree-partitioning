from __future__ import annotations
from typing import Callable

from time import perf_counter
from tree_partitioning.classes import Case, ReducedGraph
from tree_partitioning.algorithms.line_switching import milp_line_switching, brute_force
from tree_partitioning.algorithms.full._legacy import networkanalysis


def two_stage(
    n_clusters: int,
    objective: str,
    partitioning_alg: Callable[..., Partition],
    line_switching_alg: Callable[..., Solution],
    results=False,
):
    """
    Solve the tree partitioning problem with n_clusters and minimize objective.
    """
    # Initialization
    case = Case()
    net, netdict, G, igg = case.all_objects

    runtime_partition = -perf_counter()
    partition = partitioning_alg(n_clusters)
    runtime_partition += perf_counter()

    assert partition.is_connected_clusters(G)

    runtime_line_switching = -perf_counter()
    solution = line_switching_alg(partition, objective)
    runtime_line_switching += perf_counter()

    if results:
        reduced_graph = ReducedGraph(G, partition)
        # Compute partitioning results
        method_partition = partitioning_alg.__name__
        runtime_partition = runtime_partition
        n_clusters = n_clusters
        pre_bridge_block_sizes = sorted(networkanalysis(igg)[0].sizes(), reverse=True)[
            :5
        ]
        n_cross_edges = len(reduced_graph.cross_edges)
        fraction_cross_edges = n_cross_edges / len(G.edges)
        n_lines_to_be_switched_off = n_cross_edges - (n_clusters - 1)
        ms = [partition.membership[name] for name in igg.vs["name"]]
        modularity_score = igg.modularity(ms, weights="weight")

        # Compute line switching results
        method_line_switching = line_switching_alg.__name__
        runtime_line_switching = runtime_line_switching
        objective = (
            solution.model.objective()
            if method_line_switching == "milp_line_switching"
            else solution._best_gamma
        )
        n_spanning_trees = (
            solution._n_spanning_trees
            if method_line_switching == "brute_force"
            else None
        )
        _post_igg = igg.copy()
        _switched_lines = [netdict["lines"][e]["name"] for e in solution.switched_lines]
        _post_igg.delete_edges(
            [i for i, name in enumerate(igg.es["name"]) if name in _switched_lines]
        )
        post_bridge_block_sizes = sorted(
            networkanalysis(_post_igg)[0].sizes(), reverse=True
        )[:5]
        n_congested_lines = (
            len(
                [
                    (e, val)
                    for e, val in [
                        (e, v() / netdict["lines"][e]["c"])
                        for e, v in solution.model.fabs.items()
                    ]
                    if val > 1.005
                ]
            )
            if method_line_switching == "milp_line_switching"
            else sum(solution.post_switching_net.res_line.loading_percent > 100)
            + sum(solution.post_switching_net.res_trafo.loading_percent > 100)
        )

        n_lines_to_be_switched_off = n_lines_to_be_switched_off
        fraction_lines_switched_off = n_lines_to_be_switched_off / len(G.edges)

        init_gamma = (
            max(
                case.net.res_line.loading_percent.max(),
                case.net.res_trafo.loading_percent.max(),
            )
            / 100
        )

        all_results = [
            case.name,
            method_partition,
            runtime_partition,
            n_clusters,
            pre_bridge_block_sizes,
            n_cross_edges,
            fraction_cross_edges,
            n_lines_to_be_switched_off,
            modularity_score,
            method_line_switching,
            runtime_line_switching,
            n_spanning_trees,
            init_gamma,
            objective,
            post_bridge_block_sizes,
            n_congested_lines,
            n_lines_to_be_switched_off,
            fraction_lines_switched_off,
            "two_stage",
        ]

        with open("results.csv", "a") as fi:
            fi.write(";".join([str(res) for res in all_results]))
            fi.write("\n")

        # except:
        #     with open("infeasible.csv", "a") as fi:
        #         fi.write(
        #             ";".join(
        #                 str(res)
        #                 for res in [
        #                     case.name,
        #                     n_clusters,
        #                     partitioning_alg.__name__,
        #                     line_switching_alg.__name__,
        #                 ]
        #             )
        #         )

    return solution
