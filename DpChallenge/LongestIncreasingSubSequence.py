def longestIncreasingSubsequence(array):
    longest = [1]*len(array)
    maxIdx = 0
    sequence = [None]*len(array)

    for i in range(len(array)):
        currNum = array[i]
        for j in range(0, i):
            otherNum = array[j]
            if otherNum < currNum and longest[j] + 1 > longest[i]:
                longest[i] = longest[j] + 1
                sequence[i] = j

        if longest[i] > longest[maxIdx]:
            maxIdx = i

    return getSequence(sequence, maxIdx, array)

def getSequence(sequence, i, array):
    seq = []
    while i is not None:
        seq.append(array[i])
        i = sequence[i]

    return seq[::-1]
