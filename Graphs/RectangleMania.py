# Storing hash table of coords as x-y in key
def rectangleMania(coords):
    coordTable = getCoordTable(coords)
    return getRectangleCount(coords, coordTable)

def getCoordTable(coords):
    coordTable = {}
    for x,y in coords:
        coordTable[f'{x}-{y}'] = True
    return coordTable

def getRectangleCount(coords, coordTable):
    rectangleCount = 0
    for x1, y1 in coords:
        for x2, y2 in coords:
            if not isUpperRight(x1,y1,x2,y2):
                continue
            upperCoord = f'{x1}-{y2}'
            rightCoord = f'{x2}-{y1}'
            if upperCoord in coordTable and rightCoord in coordTable:
                rectangleCount += 1

    return rectangleCount

def isUpperRight(x1, y1, x2, y2):
    return x2 > x1 and y2 > y1
