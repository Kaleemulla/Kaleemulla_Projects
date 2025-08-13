def diskStacking(disks):
    # Since stack must have max, height and lower disk shld be bigger than upper
    # Sort array by 1D i.e height ASC
    disks.sort(key = lambda disk: disk[2])

    heights = [disk[2] for disk in disks] # Minimum Max height at each point is self since sorted by height
    sequences = [None for disk in disks]
    maxHeightIdx = 0

    for i in range(1, len(disks)):
        currentDisk = disks[i]
        for j in range(0, i):
            otherDisk = disks[j]
            if otherDisk[0] < currentDisk[0] and otherDisk[1] < currentDisk[1] and otherDisk[2] < currentDisk[2]: # Height check to cover if equal height then do not consider that
                if heights[i] <= (currentDisk[2] + heights[j]):
                    heights[i] = (currentDisk[2] + heights[j])
                    sequences[i] = j
        if heights[i] >= heights[maxHeightIdx]:
            maxHeightIdx = i

    return buildSequences(disks, sequences, maxHeightIdx)
    pass

def buildSequences(disks, sequences, idx):
    sequence = []

    while idx is not None:
        sequence.append(disks[idx])
        idx = sequences[idx]

    return list(reversed(sequence))
