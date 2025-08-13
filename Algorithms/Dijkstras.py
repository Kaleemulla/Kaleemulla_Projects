def dijkstrasAlgorithm(start, edges):
    numberOfVertices = len(edges)

    minDistances = [float('inf') for _ in range(numberOfVertices)]
    minDistances[start] = 0

    minHeap = MinHeap([(idx, float('inf')) for idx in range(numberOfVertices)])
    minHeap.update(start, 0)

    while not minHeap.isEmpty():
        vertex, currMin = minHeap.remove()

        if currMin == float('inf'):
            break

        for edge in edges[vertex]:
            destination, distance = edge

            newDistance = currMin + distance
            currDistance = minDistances[destination]

            if newDistance < currDistance:
                minDistances[destination] = newDistance
                minHeap.update(destination, newDistance)

    return list(map(lambda x: -1 if x == float('inf') else x, minDistances))

class MinHeap:
    def __init__(self, array):
        # Do not edit the line below.
        self.vertexMap = {idx: idx for idx in range(len(array))}
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
            if child2Idx != -1 and heap[child2Idx][1] < heap[child1Idx][1]:
                idxToSwap = child2Idx
            else:
                idxToSwap = child1Idx
            if heap[idxToSwap][1] < heap[currIdx][1]:
                self.swap(currIdx, idxToSwap, heap)
                currIdx = idxToSwap
                child1Idx = currIdx*2 + 1
            else:
                break

    def siftUp(self, currIdx, heap):
        parentIdx = (currIdx -1)//2
        while currIdx > 0 and heap[currIdx][1] < heap[parentIdx][1]:
            self.swap(currIdx, parentIdx, heap)
            currIdx = parentIdx
            parentIdx = (currIdx - 1)//2

    def remove(self):
        if self.isEmpty():
            return
            
        self.swap(0, len(self.heap)-1, self.heap)
        vertex, distance = self.heap.pop()
        self.vertexMap.pop(vertex)
        self.siftDown(0, len(self.heap)-1, self.heap)
        return vertex, distance

    def swap(self, i, j, heap):
        self.vertexMap[heap[i][0]] = j
        self.vertexMap[heap[j][0]] = i
        heap[i], heap[j] = heap[j], heap[i]

    def update(self, vertex, value):
        self.heap[self.vertexMap[vertex]] = (vertex, value)
        self.siftUp(self.vertexMap[vertex], self.heap)
