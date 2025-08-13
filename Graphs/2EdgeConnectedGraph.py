def twoEdgeConnectedGraph(edges):
    if len(edges) == 0:
        return True # Empty graph is 2-connected

    arrivalTimes = [-1] * len(edges) # If arrivaltime is -1, unvisited vertex
    startVertex = 0

    # Parent of node-0 is taken as -1, initial minArrival taken 0, can be 1
    # In recurrsion parent would be currNode, arrivalTime will be +1
    if getMinimumArrivalTimeOfAncestors(startVertex, -1, 0, arrivalTimes, edges) == -1:
        return False

    return areAllVeticesVisited(arrivalTimes)
    # if not -1, then see if all vertices were visited
    # If yes the graph started connected and is now 2-edge connected

def areAllVeticesVisited(arrivalTimes):
    for time in arrivalTimes:
        if time == -1:
            return False
    return True

def getMinimumArrivalTimeOfAncestors(currentVertex, parent, currentTime, arrivalTimes, edges):
    # currentTime is starting from 0, can also start from 1 as explained
    arrivalTimes[currentVertex] = currentTime

    minimumArrivalTime = currentTime

    for destination in edges[currentVertex]:
        if arrivalTimes[destination] == -1: # Tree edge, not visited
            minimumArrivalTime = min(minimumArrivalTime, getMinimumArrivalTimeOfAncestors
                                    (destination, currentVertex, currentTime+1, arrivalTimes, edges))
        elif destination != parent: # Not tree edge, other edge already been visited
            minimumArrivalTime = min(minimumArrivalTime, arrivalTimes[destination])
            # Just set MinArrival to min of arrival at destin and minArrival
            # We reach here after getting value from IF recurrsion
            # If we dont reach here, no edge to parent and minArrival will be inital value

    if minimumArrivalTime == currentTime and parent != -1: # parent != -1 to skip first node on this condition
        return -1

    return minimumArrivalTime
