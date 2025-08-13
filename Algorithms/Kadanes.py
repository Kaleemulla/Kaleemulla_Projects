def kadanesAlgorithm(array):
    maxEndingHere = float('-inf')
    maxSoFar = float('-inf')

    for num in array:
        maxEndingHere = max(maxEndingHere+num, num)
        maxSoFar = max(maxSoFar, maxEndingHere)

    return maxSoFar
