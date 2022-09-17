def _single_stage_warm_start(case, generators, **kwargs):
    """
    Use two-stage to warm-start single stage.
    """
    solver, options = kwargs["solver"], kwargs["options"]

    G = case.G

    model1 = partitioning.power_flow_disruption(G, generators, **kwargs)
    solver.solve(model1, tee=False, options=options)

    partition = model_utils.get_partition(model1)
    rg = ReducedGraph(G, partition).RG.to_undirected()
    cost, lines = maximum_spanning_tree(G, partition)

    _sanity_check(G, generators, partition, lines)
    print("two_stage", len(generators), cost)
