def stringsMadeUpOfStrings(strings, substrings):
    trie = Trie()
    for substring in substrings:
        trie.insert(substring)

    solutions = []
    for string in strings:
        if isMadeUpOfStrings(string, 0, trie, {}):
            solutions.append(string)

    return solutions

def isMadeUpOfStrings(string, start, trie, memo):
    if start == len(string):
        return True
    if start in memo:
        return memo[start]

    curr = trie.root
    for i in range(start, len(string)):
        ch = string[i]
        if ch not in curr:
            break

        curr = curr[ch]
        if curr["isEndOfString"] and isMadeUpOfStrings(string, i+1, trie, memo):
            memo[start] = True
            return True
            
    memo[start] = False

class Trie:
    def __init__(self):
        self.root = {"isEndOfString": False}

    def insert(self, string):
        curr = self.root

        for i in range(len(string)):
            if string[i] not in curr:
                curr[string[i]] = {"isEndOfString": False}
            curr = curr[string[i]]
        curr["isEndOfString"] = True
