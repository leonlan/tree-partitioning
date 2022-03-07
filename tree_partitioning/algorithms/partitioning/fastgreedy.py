#!/usr/bin/env ipython
def fastgreedy(igg, n_clusters, weights):
    """
    Cluster the graph using fastgreedy
    """
    # The induced subgraph corresponding to the biggest bridge-block gets split into n_clusters
    igg.simplify(combine_edges=dict(weight="sum"))
    sgpartition = sg.community_fastgreedy(weights="weight").as_clustering(
        n=n_clusters
    )

    # Cluster sizes and modularity score of the selected partition of sg
    bsplit_pre = list(map(len, list(sgpartition)))
    Q = sg.modularity(sgpartition, weights="weight")

# The original graph partition gets updated accordingly
        # TODO: Understand this part. In what sense to the partitions gets updated?
        newm = block_partition.membership  # new membership vector for the full graph
        sgcommunities = []
        sgpm = np.array(sgpartition.membership)
        for c in range(n_clusters):
            if (
                c == 0
            ):  # For the first cluster in sg, leave the membership unchanged in newm and store the value in sg.communities
                idxcommunity = np.where(sgpm == c)[0]
                sgcommunities.append(
                    newm[G.vs.select(name_eq=sg.vs["name"][idxcommunity[0]])[0].index]
                )
            else:
                idxcommunity = np.where(sgpm == c)[
                    0
                ]  # find indices of nodes that belong to community c
                newcommunityindex = (
                    max(newm) + 1
                )  # community c inside subgraph gets updated index, which is equal to the current max index +1
                sgcommunities.append(newcommunityindex)
                for idx in [sg.vs["name"][sgidx] for sgidx in idxcommunity]:
                    newm[
                        G.vs.select(name_eq=idx)[0].index
                    ] = newcommunityindex  # all nodes that belong to community c gets updated memberhip in the full graph partition
        G.vs["community"] = newm
        npartition = ig.VertexClustering(
            G, membership=newm
        )  # new partition for the full graph
        bsize_pre = [npartition.subgraph(i).vcount() for i in range(len(npartition))]

        # The following for loop deals with the selected partition possibly having disconnected components
        connectedpartition_flag = True
        for ip in range(len(npartition)):
            if not (npartition.subgraph(ip).is_connected()):
                connectedpartition_flag = False
                for k in range(1, len(npartition.subgraph(ip).clusters())):
                    newcommunityindex = max(newm) + 1
                    sgcommunities.append(newcommunityindex)
                    v = [
                        npartition.subgraph(ip).vs["name"][j]
                        for j in npartition.subgraph(ip).clusters()[k]
                    ]
                    for idx in v:
                        newm[G.vs.select(name_eq=idx)[0].index] = newcommunityindex
                        G.vs.select(name_eq=idx)[0]["community"] = newcommunityindex
        if not (connectedpartition_flag):
            #             print('Flag: partition with disconnected clusters')
            G.vs["community"] = newm
            npartition = ig.VertexClustering(G, membership=newm)

        # final partition is casted into a partition of the original graph igg
        npartition = ig.VertexClustering(igg, membership=newm)
        bsize_post = [npartition.subgraph(i).vcount() for i in range(len(npartition))]

        # color attribute is given to the graph nodes depending on the cluster they belong to
        igg.vs["community"] = newm
        igg.vs["color"] = ig.drawing.colors.ClusterColoringPalette(
            len(npartition)
        ).get_many(npartition.membership)
