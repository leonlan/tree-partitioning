import pyomo.environ as pyo


def _reduced_graph_tree(m):
    """
    Activates at most k - 1 cross edges, where k is the number of clusters.
    Together with some connectivity constraints (e.g., single commodity flow)
    this ensures that the spanning tree is a reduced graph.
    """
    m.active_cross_edge = pyo.Var(m.lines, domain=pyo.Binary)

    @m.Constraint(m.lines)
    def relate_assignment_to_activeness(m, *line):
        rhs1 = sum(m.assign_line[line, cluster] for cluster in m.clusters)
        rhs2 = m.active_cross_edge[line]
        lhs = m.active_line[line]
        return rhs1 + rhs2 == lhs

    @m.Constraint()
    def exactly_k_minus_1_cross_edges(m):
        return sum(m.active_cross_edge[line] for line in m.lines) == len(m.clusters) - 1
