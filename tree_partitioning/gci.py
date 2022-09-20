from collections import defaultdict

import networkx as nx


def mst_gci(case, n_clusters: int, _SPLIT=0.15):
    """
    Find generator groups using a minimum spanning tree algorithm.

    The identified MST is split iteratively such that each subtree contains
    a generator group of reasonable size. This is at least _SPLIT fraction
    per cluster.
    """
    G = nx.Graph(case.G)
    neg = {e: {"neg_weight": -G.edges[e]["weight"]} for e in G.edges}
    nx.set_edge_attributes(G, neg)  # TODO do testing
    T = nx.tree.minimum_spanning_tree(G, weight="neg_weight")

    gen_idcs = get_gen_idcs(case.net)

    # Find which edges need to be removed from the tree such that each
    # subtree is split roughly in half.
    for _ in range(n_clusters - 1):
        components = [cc for cc in nx.algorithms.connected_components(T)]
        largest_cc = nx.Graph(T.subgraph(max(components, key=len)))
        cc_gens = sum([1 for gen in gen_idcs if gen in largest_cc])

        # TODO select edge with best_ratio
        for edge in largest_cc.edges:
            largest_cc.remove_edge(*edge)
            block0 = [cc for cc in nx.algorithms.connected_components(largest_cc)][0]
            n_gens = sum([1 for gen in gen_idcs if gen in block0])

            if _SPLIT <= n_gens / cc_gens <= 1 - _SPLIT:
                # Remove edge from the tree if the component is of good size
                T.remove_edge(*edge)
                break
            else:
                # Add edge back and go to the next edge
                largest_cc.add_edge(*edge)

    # Retrieve the generators from each component
    components = [cc for cc in nx.algorithms.connected_components(T)]
    generator_groups = defaultdict(list)

    for idx in gen_idcs:
        for cluster_idx, component in enumerate(components):
            if idx in component:
                generator_groups[cluster_idx].append(idx)

    return generator_groups


def get_gen_idcs(net):
    """
    Return the bus indices of generators.
    """
    # NOTE Use poly cost as proxy for being generator or not
    # because sometimes SGENS are included in gens (e.g., see IEEE-118)
    pc = net.poly_cost

    # NOTE do we also want to check for cost to filter non-generators??
    # gens = pc.loc[(pc.cp1_eur_per_mw > 0) & (pc.et == "gen"), "element"]
    gens = pc.loc[pc.et == "gen", "element"]

    # Ensure that the index is always feasible
    gens = list(set(gens.values) & set(net.gen.index.values))
    gen_idcs = net.gen.loc[gens].bus.values
    return gen_idcs
