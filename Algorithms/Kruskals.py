def kruskalsAlgorithm(edges):
    # input given 2 way adjacency list, [edge, weight]
    # enumerate gives node number
    # skip reverse node edge to consider only one way edge
    edgeList = []

    for vertex, edgeWeightPair in enumerate(edges):
        for edge in edgeWeightPair:
            if edge[0] > vertex: # consider only one way edge
                edgeList.append([vertex, edge[0], edge[1]])

    sortedEdges = sorted(edgeList, key=lambda edge: edge[2])
    # [vertex, vertexEdge, weight]

    parents = list(range(len(edges))) # instead of {} using [] with idx as vertex #
    ranks = [0]*len(edges) # initialize all ranks as 0
    mst = [[] for _ in range(len(edges))]

    for edge in sortedEdges:
        vertex1Root = find(edge[0], parents)
        vertex2Root = find(edge[1], parents)
        if vertex1Root != vertex2Root: # nodes are not connected/visited
            mst[edge[0]].append([edge[1], edge[2]])
            mst[edge[1]].append([edge[0], edge[2]])
            union(vertex1Root, vertex2Root, parents, ranks) # connect nodes

    return mst

def find(vertex, parents):
    if vertex != parents[vertex]:
        parents[vertex] = find(parents[vertex], parents)

    return parents[vertex]

def union(vertex1Root, vertex2Root, parents, ranks):
    if ranks[vertex1Root] < ranks[vertex2Root]:
        parents[vertex1Root] = vertex2Root
    elif ranks[vertex1Root] > ranks[vertex2Root]:
        parents[vertex2Root] = vertex1Root
    else:
        parents[vertex2Root] = vertex1Root
        ranks[vertex1Root] += 1
