# T = O(n+m), S = O(m)
def knuthMorrisPrattAlgorithm(string, substring):
    pattern = buildPattern(substring)
    return doesMatch(string, substring, pattern)

def buildPattern(substring):
    pattern = [-1]*len(substring)
    j = 0
    i = 1

    while i < len(substring):
        if substring[i] == substring[j]:
            pattern[i] = j
            i += 1
            j += 1
        elif j > 0: # isnt equal but j>0, might have pattern
            j = pattern[j-1]+1
        else:
            i += 1

    return pattern

def doesMatch(string, substring, pattern):
    i = 0
    j = 0

    # while i < len(string):
    while i+len(substring)-j <= len(string): # stop optimization even before reaching j=end
        if string[i] == substring[j]:
            if j == len(substring) - 1:
                return True
            i += 1
            j += 1
        elif j > 0:
            j = pattern[j-1]+1
        else:
            i += 1
            
    return False
        
