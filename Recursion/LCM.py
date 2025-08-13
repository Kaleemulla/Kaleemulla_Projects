def getLowestCommonManager(topManager, reportOne, reportTwo):
    return getOrgInfo(topManager, reportOne, reportTwo).lowestCM

def getOrgInfo(manager, reportOne, reportTwo):
    numImpReports = 0

    for directReport in manager.directReports:
        orgInfo = getOrgInfo(directReport, reportOne, reportTwo)
        if orgInfo.lowestCM is not None:
            return orgInfo
        numImpReports += orgInfo.numImpReports

    if manager ==  reportOne or manager == reportTwo: # When manager node itself is report (leaf)
        numImpReports += 1

    lowestCM = manager if numImpReports == 2 else None

    return OrgInfo(lowestCM, numImpReports)

class OrgInfo:
    def __init__(self, lowestCM, numImpReports):
        self.lowestCM = lowestCM
        self. numImpReports = numImpReports
        
# This is an input class. Do not edit.
class OrgChart:
    def __init__(self, name):
        self.name = name
        self.directReports = []

# O(N) time, N is no. of nodes
# O(D) space, D is depth of tree
