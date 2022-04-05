from tree_partitioning.classes import Case, Partition, Solution


def powerset(iterable):
    """Computes the powerset of an iterable.

    Example:
    - powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def EP2E(edge_P_ids, G):
    """Maps edges from EP to E."""
    mapper = dict()
    name2e = {G.get_edge_data(*e)["name"]: e for e in G.edges}
    for u, v, w in edge_P_ids:
        mapper[(u, v, w)] = name2e[w]
    return mapper


def obs_brute_force(G, net, P, acpf=True, ac_options={}):
    """Brute force the Optimal Bridge Selection problem.

    :param G: networkx graph
    :param net: pandapower network
    :param P: partition
    :param acpf: boolean to consider AC power flow equations
    """

    def _is_edges_spanning_tree(edges):
        """Checks if the k-1 edges form a spanning tree."""
        K = nx.Graph()
        K.add_edges_from(edges)
        return nx.is_tree(K)

    rg = reduced_graph(G, P)

    # Describe the results
    res = defaultdict(list)

    # Step 1: Compute the spanning tree cross-edge combinations
    # by considering the simple reduced graph [subtour elimination]
    H = nx.Graph(rg)
    feasible_combinations = []
    for edges in combinations(H.edges, r=P.size - 1):
        if _is_edges_spanning_tree(edges):
            feasible_combinations.append(edges)
    # Step 2: Compute the product of all cross-edges that belong
    # to the feasible set of combinations
    edge2name = defaultdict(list)
    for e in rg.edges:
        i, j = e[0], e[1]
        name = rg.get_edge_data(*e)["name"]
        edge2name[(i, j)].append(name)

    all_lines = list(chain(*edge2name.values()))
    for edges in feasible_combinations:
        # Lines are tuples of line names that we keep
        for lines in product(*list(map(edge2name.get, edges))):
            # Deactive all other lines
            other_lines = [l for l in all_lines if l not in lines]
            net_post = deactivate_lines_pp(net, other_lines)
            G_post = deactivate_lines_nx(G, other_lines)
            assert verify_bbd(G_post, P)

            # Calculate the power flow disruption
            pfd = sum(
                [
                    G.get_edge_data(*e)["weight"]
                    for e in G.edges
                    if G.get_edge_data(*e)["name"] in other_lines
                ]
            )
            res["power_flow_disruption"].append(pfd)

            # Calculate DC congestion
            try:
                pp.rundcpp(net_post)
                gamma_dc = max_loading_percent(net_post)
            except:  # TODO: Which error should be put here?
                gamma_dc = np.isnan
            res["gamma_dc"].append(gamma_dc)

            # Calculate AC congestion
            if acpf:
                try:
                    pp.runpp(net_post)
                    gamma_ac = max_loading_percent(net_post)
                    print(gamma_ac)
                    # run_julia_acpf(net_post)
                except pp.LoadflowNotConverged:
                    print("ACPF did not converge.")
                    gamma_ac = np.nan
                res["gamma_ac"].append(gamma_ac)

            # print(lines)
    return res


def brute_force(partition: Partition):
    """
    Solves the Line Switching Problem using brute force
    and returns the corresponding Tree Partition.
    """
    pass
