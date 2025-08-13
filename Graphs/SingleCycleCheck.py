def hasSingleCycle(array):
    visited = 0
    i = 0
    while visited < len(array):
        if visited > 0 and i == 0:
            return False
        visited += 1
        i = getNextIdx(i, array)

    return i == 0

def getNextIdx(i, array):
    jump = array[i]
    nextIdx = (i + jump) % len(array)
    return nextIdx
