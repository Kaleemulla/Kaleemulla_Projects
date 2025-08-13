def maxSumIncreasingSubsequence(array):
    sums = array[:]
    sequences = [None for i in array]
    maxSumIdx = 0

    for i in range(len(array)):
        currentNum = array[i]
        for j in range(0, i):
            otherNum = array[j]
            if otherNum < currentNum and (sums[j] + currentNum) >= sums[i]: #Strictly increasing <, if include same then <=. If >= then you are using last max item seq, if just > then first item (ask both to interviewer)
                sums[i] = sums[j] + currentNum
                sequences[i] = j
                
        if sums[i] >= sums[maxSumIdx]:
            maxSumIdx = i

    return [sums[maxSumIdx], buildSequence(array, sequences, maxSumIdx)]
    pass

def buildSequence(array, sequences, idx):
    sequence = []

    while idx is not None:
        sequence.append(array[idx])
        idx = sequences[idx]

    return list(reversed(sequence))
