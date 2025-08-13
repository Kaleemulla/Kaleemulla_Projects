class Node:
    def __init__(self, row, col, value):
        self.id = str(row) + '-' + str(col)
        self.row = row
        self.col = col
        self.value = value
        self.g = float('inf')
        self.f = float('inf')
        self.cameFrom = None
        
def aStarAlgorithm(startRow, startCol, endRow, endCol, graph):
    nodes = initializeNodes(graph)

    startNode = nodes[startRow][startCol]
    endNode = nodes[endRow][endCol]

    startNode.g = 0
    startNode.f = calculateManhattanDistance(startNode, endNode)

    nodesToVisit = MinHeap([startNode]) # you can replace nodesToVisit with minHeap

    while not nodesToVisit.isEmpty():
        currNode = nodesToVisit.remove()

        if currNode == endNode:
            break

        neighbors = getNeighbors(currNode, nodes)

        for neighbor in neighbors:
            if neighbor.value == 1: # Obstacle
                continue

            tentativeNeighborG = currNode.g + 1

            if tentativeNeighborG >= neighbor.g: # means visited node
                continue

            neighbor.cameFrom = currNode
            neighbor.g = tentativeNeighborG
            neighbor.f = (
                tentativeNeighborG + 
                calculateManhattanDistance(neighbor, endNode)
            )

            if not nodesToVisit.containsNode(neighbor):
                nodesToVisit.insert(neighbor)
            else:
                nodesToVisit.update(neighbor)

    return reconstructPath(endNode)

def getNeighbors(node, nodes):
    neighbors = []

    numRows = len(nodes)
    numCols = len(nodes[0])

    row = node.row
    col = node.col

    if row < numRows -1:
        neighbors.append(nodes[row+1][col])
    if row > 0:
        neighbors.append(nodes[row-1][col])
    if col < numCols -1:
        neighbors.append(nodes[row][col+1])
    if col > 0:
        neighbors.append(nodes[row][col-1])

    return neighbors
    
def reconstructPath(endNode):
    if not endNode.cameFrom:
        return []

    currNode = endNode
    path = []

    while currNode is not None:
        path.append([currNode.row, currNode.col])
        currNode = currNode.cameFrom

    return path[::-1]
def initializeNodes(graph):
    nodes = []

    for i, row in enumerate(graph):
        nodes.append([])
        for j, value in enumerate(row):
            nodes[i].append(Node(i, j, value))

    return nodes

def calculateManhattanDistance(currNode, endNode):
    currRow = currNode.row
    currCol = currNode.col
    endRow = endNode.row
    endCol = endNode.col

    return abs(currRow - endRow) + (currCol - endCol)

class MinHeap:
    def __init__(self, array):
        # Do not edit the line below.
        self.nodesPositionsInHeap = {node.id: idx for idx,node in enumerate(array)}
        self.heap = self.buildHeap(array)

    def isEmpty(self):
        return len(self.heap) == 0

    def buildHeap(self, array):
        firstParentIdx = (len(array)-2)//2 #lastidx-1 = len-2
        for currIdx in reversed(range(firstParentIdx+1)):
            self.siftDown(currIdx, len(array)-1, array)

        return array

    def siftDown(self, currIdx, endIdx, heap):
        child1Idx = currIdx*2 +1
        while child1Idx <= endIdx:
            child2Idx = currIdx*2 + 2 if currIdx*2 + 2 <= endIdx else -1
            if child2Idx != -1 and heap[child2Idx].f < heap[child1Idx].f:
                idxToSwap = child2Idx
            else:
                idxToSwap = child1Idx
            if heap[idxToSwap].f < heap[currIdx].f:
                self.swap(currIdx, idxToSwap, heap)
                currIdx = idxToSwap
                child1Idx = currIdx*2 + 1
            else:
                break

    def siftUp(self, currIdx, heap):
        parentIdx = (currIdx -1)//2
        while currIdx > 0 and heap[currIdx].f < heap[parentIdx].f:
            self.swap(currIdx, parentIdx, heap)
            currIdx = parentIdx
            parentIdx = (currIdx - 1)//2

    def peek(self):
        return self.heap[0]

    def remove(self):
        if self.isEmpty():
            return
            
        self.swap(0, len(self.heap)-1, self.heap)
        node = self.heap.pop()
        del self.nodesPositionsInHeap[node.id]
        self.siftDown(0, len(self.heap)-1, self.heap)
        return node

    def insert(self, node):
        self.heap.append(node)
        self.nodesPositionsInHeap[node.id] = len(self.heap) - 1
        self.siftUp(len(self.heap)-1, self.heap)

    def swap(self, i, j, heap):
        self.nodesPositionsInHeap[heap[i].id] = j
        self.nodesPositionsInHeap[heap[j].id] = i
        heap[i], heap[j] = heap[j], heap[i]

    def containsNode(self, node):
        return node.id in self.nodesPositionsInHeap

    def update(self, node):
        self.siftUp(self.nodesPositionsInHeap[node.id], self.heap)
