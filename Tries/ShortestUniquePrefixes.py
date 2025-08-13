def shortestUniquePrefixes(strings):
    trie = Trie()

    for string in strings:
        trie.insert(string)

    prefixes = []
    for string in strings:
        uniquePrefix = findUniquePrefix(string, trie)
        prefixes.append(uniquePrefix)

    return prefixes

def findUniquePrefix(string, trie):
    curr = trie.root
    i = 0

    while i < len(string) -1:
        ch = string[i]
        curr = curr[ch]
        if curr["count"] == 1:
            break

        i += 1

    return string[0:i+1]

class Trie:
    def __init__(self):
        self.root = {"count": 0}

    def insert(self, string):
        curr = self.root

        for i in range(len(string)):
            if string[i] not in curr:
                curr[string[i]] = {"count": 0}
            curr = curr[string[i]]
            curr["count"] += 1
