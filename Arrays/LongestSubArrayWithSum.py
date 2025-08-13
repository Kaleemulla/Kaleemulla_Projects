def longestSubarrayWithSum(array, targetSum):
    indices = []
    currentsum = 0
    startidx = 0
    endidx = 0

    while endidx < len(array): # Keep moving only ending idx
        currentsum += array[endidx]

        # Move startidx and reduce its value from currentsum, until currentsum <= targetSum
        while startidx < endidx and currentsum > targetSum:
            currentsum -= array[startidx]
            startidx += 1

        if currentsum == targetSum: # If targetSum, compare with existing indices
            if len(indices) == 0 or indices[1] - indices[0] < endidx - startidx:
                indices = [startidx, endidx]
                
        endidx += 1
    
    return indices
