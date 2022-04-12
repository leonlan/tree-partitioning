import igraph as ig
import pandas as pd


def _igg_from_netdict(netdict):
    """
    Create an igraph graph from netdict.
    """
    igg = ig.Graph.TupleList(
        pd.DataFrame.from_dict(netdict["lines"]).T.itertuples(index=False),
        directed=False,
        vertex_name_attr="name",
        weights=False,
        edge_attrs=[
            "in_service",
            "name",
            "f",
            "weight",
            "loading_percent",
            "type",
            "index_by_type",
            "edge_index",
            "b",
            "c",
            "edge_id",
        ],
    )
    igg.vs["community"] = [0] * (igg.vcount())
    igg.vs["p"] = pd.DataFrame(netdict["buses"]).T.p_mw

    return igg
