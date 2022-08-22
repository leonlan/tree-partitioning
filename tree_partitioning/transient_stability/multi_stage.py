def multi_stage(case, generators, time_limit):
    G = G.deepcopy()
    P = ...
    E = ...

    for r in range(k):
        # Take the largest partition
        largest = max(P, key=len)

        # Consider subgraph
        subgraph = G.subgraph[largest]

        # Solve TPI
        V, W = tpi(subgraph, k=2)

        # Solve OLS on
        rg = reduced_graph(G, P=(V, W))
        removed = ols(G, rg, ...)

        # Add new clusters, remove old cluster
        P.remove(largest)
        P.add(V)
        P.add(W)

        # Remove lines
        G.remove(removed)
        E.add(removed)

    return G, P, E, ...
