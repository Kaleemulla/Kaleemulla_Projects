#T,S = O(w*h)
def removeIslands(matrix):
    BorderOnes = [[False for col in matrix[0]] for row in matrix]

    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            rowIsBorder = row == 0 or row == len(matrix) -1 
            colIsBorder = col == 0 or col == len(matrix[row]) - 1
            isBorder = rowIsBorder or colIsBorder

            if isBorder and matrix[row][col] == 1:
                findOnesConnectedBorder(matrix, row, col, BorderOnes)

    for row in range(1, len(matrix)-1):
        for col in range(1, len(matrix[row])-1):
            if BorderOnes[row][col]:
                continue
            matrix[row][col] = 0

    return matrix

def findOnesConnectedBorder(matrix, i, j, BorderOnes):
    stack = [[i, j]]

    while len(stack) > 0:
        curr = stack.pop()
        i, j = curr
        visited = BorderOnes[i][j]
        
        if visited:
            continue
            
        BorderOnes[i][j] = True

        neighbors = getNeighbors(matrix, i, j)

        for neighbor in neighbors:
            row, col = neighbor

            if matrix[row][col] != 1:
                continue

            stack.append(neighbor)

def getNeighbors(matrix, row, col):
    neighbors = []

    if row - 1 >= 0:
        neighbors.append([row-1, col])
    if row + 1 < len(matrix):
        neighbors.append([row+1, col])
    if col -1 >= 0:
        neighbors.append([row, col-1])
    if col + 1 < len(matrix[row]):
        neighbors.append([row, col+1])

    return neighbors
        
