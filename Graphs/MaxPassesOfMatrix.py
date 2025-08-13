# T,S = O(wxh)
def minimumPassesOfMatrix(matrix):
    # Instead of looking at -ve and find if they can be made +ve
    # We see all position of +ve and convert neighbors to -ve
    # Do this over queue, discard old queue and create new with new +ve value
    # Do until new queue is empty
    passes = convertNegative(matrix)
    return passes - 1 if not containsNegative(matrix) else -1

def convertNegative(matrix):
    queue = getAllPositive(matrix)
    passes = 0

    while len(queue): # Check solution-2 that uses 1 queue
        currentQueue = queue
        queue = []

        while currentQueue:
            i, j = currentQueue.pop(0) #O(n), use std queue for O(1)

            neighbors = getNeighbors(i, j, matrix)
            for neighbor in neighbors:
                row, col = neighbor
                value = matrix[row][col]
                if value < 0:
                    matrix[row][col] = value*-1
                    queue.append([row, col])
        passes += 1
    return passes

def getAllPositive(matrix):
    positions = []
    
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] > 0:
                positions.append([i, j])

    return positions

def getNeighbors(row, col, matrix):
    neighbors = []

    if row > 0:
        neighbors.append([row-1, col])
    if row < len(matrix) - 1:
        neighbors.append([row+1, col])
    if col > 0:
        neighbors.append([row, col-1])
    if col < len(matrix[0]) -1:
        neighbors.append([row, col+1])

    return neighbors

def containsNegative(matrix):
    for row in matrix:
        for value in row:
            if value < 0:
                return True

    return False
    
