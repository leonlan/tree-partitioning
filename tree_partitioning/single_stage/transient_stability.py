import pyomo.environ as pyo

from ._base_model import _base_model


def transient_stability(case, generators, time_limit):
    """
    Solve the tree partitioning problem for transient stability using the
    single-stage MILP approach.
    """
    m = _base_model(case, generators)

    @m.Objective(sense=pyo.minimize)
    def objective(m):
        return sum(m.line_data[e]["f"] * (1 - m.active_line[e]) for e in m.lines)

    # Solve
    solver = pyo.SolverFactory("gurobi", solver_io="python")

    options = {}
    options["TimeLimit"] = time_limit

    res = solver.solve(m, tee=False, options=options)
    print(f"test: {m.objective()}")

    # # Print solution
    # print(
    #     f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"
    # )

    return m, res
