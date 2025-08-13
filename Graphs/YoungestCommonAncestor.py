class AncestralTree:
    def __init__(self, name):
        self.name = name
        self.ancestor = None

#T = O(d) and S = O(1)
def getYoungestCommonAncestor(topAncestor, descendantOne, descendantTwo):
    D1 = getDepth(descendantOne, topAncestor)
    D2 = getDepth(descendantTwo, topAncestor)

    if D1 > D2:
        return backTrackAnsTree(descendantOne, descendantTwo, D1-D2)
    else:
        return backTrackAnsTree(descendantTwo, descendantOne, D2-D1)        

def getDepth(desc, top):
    depth = 0
    while desc != top:
        depth += 1
        desc = desc.ancestor

    return depth

def backTrackAnsTree(lowerDesc, higherDesc, diff):
    while diff > 0:
        lowerDesc = lowerDesc.ancestor
        diff -=1

    while lowerDesc != higherDesc:
        lowerDesc = lowerDesc.ancestor
        higherDesc = higherDesc.ancestor

    return lowerDesc
