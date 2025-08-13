#T = O(w^2xh^2), S = O(wxh)
def largestIsland(matrix):
    maxSize = 0
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if matrix[row][col] == 0:
                continue
            maxSize = max(maxSize, getSizeFromNode(row, col, matrix))
    return maxSize
    # Use solution-2 it is more optimal

def getSizeFromNode(row, col, matrix):
    size = 1
    visited = [[False for value in row] for row in matrix]

    nodesToExplore = getLandNeighbors(row, col, matrix)

    while len(nodesToExplore):
        currNode = nodesToExplore.pop()
        currRow, currCol = currNode

        if visited[currRow][currCol]:
            continue

        visited[currRow][currCol] = True
        size += 1

        nodesToExplore += getLandNeighbors(currRow, currCol, matrix)

    return size

def getLandNeighbors(row, col, matrix):
    neighbors = [] # These are land neighbors, so check if != 1

    if row>0 and matrix[row-1][col] != 1:
        neighbors.append([row-1, col])
    if row<len(matrix)-1 and matrix[row+1][col] != 1:
        neighbors.append([row+1, col])
    if col>0 and matrix[row][col-1] != 1:
        neighbors.append([row, col-1])
    if col<len(matrix[0])-1 and matrix[row][col+1] != 1:
        neighbors.append([row, col+1])

    return neighbors
