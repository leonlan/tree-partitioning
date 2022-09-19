import tree_partitioning.milp.utils as model_utils


def _select_partitioning(G, generators, recursive=False, **kwargs):
    solver, options = kwargs["solver"], kwargs["options"]

    if "partitioning_model" in kwargs and (part_model := kwargs["partitioning_model"]):
        model = part_model(G, generators, recursive, **kwargs)
        solver.solve(model, tee=False, options=options)
        return model_utils.get_partition(model)

    # NOTE This is untested since we only use partitioning models
    elif "partitioning_alg" in kwargs and (part_alg := kwargs["partitioning_alg"]):
        return part_alg(G, generators, recursive, **kwargs)

    else:
        raise ValueError("No partitioning model or algorithm provided.")


def _select_line_switching(G, partition, **kwargs):
    solver, options = kwargs["solver"], kwargs["options"]

    if "line_switching_model" in kwargs and (
        ls_model := kwargs["line_switching_model"]
    ):
        model = ls_model(G, partition, **kwargs)
        solver.solve(model, tee=False, options=options)
        cost, lines = model.objective(), model_utils.get_inactive_lines(model)
        return cost, lines

    elif "line_switching_alg" in kwargs and (ls_alg := kwargs["line_switching_alg"]):
        return ls_alg(G, partition, **kwargs)

    else:
        raise ValueError("No partitioning model or algorithm provided.")
