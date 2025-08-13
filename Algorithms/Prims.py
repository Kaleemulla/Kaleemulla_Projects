def primsAlgorithm(edges):
    # Heap node format [vertex, edge, weight]
    minHeap = MinHeap([[0, edge[0], edge[1]] for edge in edges[0]])

    mst = [[] for _ in range(len(edges))]
    while not minHeap.isEmpty():
        vertex, discoveredVertex, distance = minHeap.remove()
    
        if len(mst[discoveredVertex]) == 0: # if not 0, already connected/visited
            mst[vertex].append([discoveredVertex, distance])
            mst[discoveredVertex].append([vertex, distance])
        
            for neighbor, neighborDistance in edges[discoveredVertex]:
                if len(mst[neighbor]) == 0:
                    minHeap.insert([discoveredVertex, neighbor, neighborDistance])

    return mst

class MinHeap:
    def __init__(self, array):
        # Do not edit the line below.
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
            if child2Idx != -1 and heap[child2Idx][2] < heap[child1Idx][2]:
                idxToSwap = child2Idx
            else:
                idxToSwap = child1Idx
            if heap[idxToSwap][2] < heap[currIdx][2]:
                self.swap(currIdx, idxToSwap, heap)
                currIdx = idxToSwap
                child1Idx = currIdx*2 + 1
            else:
                return

    def siftUp(self, currIdx, heap):
        parentIdx = (currIdx -1)//2
        while currIdx > 0 and heap[currIdx][2] < heap[parentIdx][2]:
            self.swap(currIdx, parentIdx, heap)
            currIdx = parentIdx
            parentIdx = (currIdx - 1)//2

    def peek(self):
        return self.heap[0]

    def remove(self):
        self.swap(0, len(self.heap)-1, self.heap)
        valueRemoved = self.heap.pop()
        self.siftDown(0, len(self.heap)-1, self.heap)
        return valueRemoved

    def insert(self, value):
        self.heap.append(value)
        self.siftUp(len(self.heap)-1, self.heap)

    def swap(self, i, j, heap):
        heap[i], heap[j] = heap[j], heap[i]
