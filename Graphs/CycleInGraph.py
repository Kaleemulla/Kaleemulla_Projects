def cycleInGraph(edges):
    num = len(edges)
    visited = [False for _ in range(num)]
    inStack = [False for _ in range(num)]

    for node in range(num):
        if visited[node]:
            continue

        containsCycle = isNodeInCycle(edges, node, visited, inStack)
        if containsCycle:
            return True

    return False

def isNodeInCycle(edges, node, visited, inStack):
    visited[node] = True
    inStack[node] = True

    neighbors = edges[node] # No need to find neighbors, already given

    for neighbor in neighbors:
        if not visited[neighbor]:
            containsCycle = isNodeInCycle(edges, neighbor, visited, inStack)

            if containsCycle:
                return True
        elif inStack[neighbor]:
            return True
    inStack[node] = False
    return False
