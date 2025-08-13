def interweavingStrings(one, two, three):
    if len(three) != len(one) + len(two):
        return False
        
    return findInterweaving(one, two, three, 0, 0)

def findInterweaving(one, two, three, i, j):
    k = i + j

    if k == len(three):
        return True

    if i < len(one) and one[i] == three[k]:
        if findInterweaving(one, two, three, i+1, j):
            return True

    if j < len(two) and two[j] == three[k]:
        return findInterweaving(one, two, three, i, j+1)

    return False
