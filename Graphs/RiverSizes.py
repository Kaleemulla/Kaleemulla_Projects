def riverSizes(matrix):
    sizes = []
    visited = [[False for _ in row] for row in matrix]

    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if visited[row][col]:
                continue
            traverseNode(row, col, matrix, visited, sizes)

    return sizes

def traverseNode(i, j, matrix, visited, sizes):
    currentRiverSize = 0
    nodesToExplore = [[i, j]] # This is a DFS stack
    
    while len(nodesToExplore):
        currentNode = nodesToExplore.pop() # Pop each from DFS stack, it keeps filling from line #32
        i, j = currentNode

        if visited[i][j]:
            continue # Double check validation

        visited[i][j] = True
        if matrix[i][j] == 0:
            continue
            
        currentRiverSize += 1
        nodesToExplore += getUnvisitedNeighbors(i, j, matrix, visited)

    if currentRiverSize > 0:
        sizes.append(currentRiverSize)

def getUnvisitedNeighbors(row, col, matrix, visited):
    unvisitedNeighbors = []
    if row - 1 >= 0 and not visited[row-1][col]:
        unvisitedNeighbors.append([row-1, col])
    if row + 1 < len(matrix) and not visited[row+1][col]:
        unvisitedNeighbors.append([row+1, col])
    if col -1 >= 0 and not visited[row][col-1]:
        unvisitedNeighbors.append([row, col-1])
    if col + 1 < len(matrix[0]) and not visited[row][col+1]:
        unvisitedNeighbors.append([row, col+1])

    return unvisitedNeighbors
