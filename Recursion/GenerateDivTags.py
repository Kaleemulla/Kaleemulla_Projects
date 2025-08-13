def generateDivTags(numberOfTags):
    result = []
    generateDivs("", numberOfTags, numberOfTags, result)
    return result

def generateDivs(prefix, opening, closing, result):
    if opening > 0:
        generateDivs(prefix+"<div>", opening-1, closing, result)

    if closing > opening:
        generateDivs(prefix+"</div>", opening, closing-1, result)

    if closing == 0:
        result.append(prefix)
