def longestMostFrequentPrefix(strings):
    trie = Trie()
    for string in strings:
        trie.insert(string)

    return trie.maxPrefixFullString[0:trie.maxPrefixLength]

class Trie:
    def __init__(self):
        self.root = {"count": 0}
        self.maxPrefixCount = 0
        self.maxPrefixLength = 0
        self.maxPrefixFullString = ""

    def insert(self, string):
        curr = self.root
        for i in range(len(string)):
            if string[i] not in curr:
                curr[string[i]] = {"count": 0}
            curr = curr[string[i]]
            curr["count"] += 1

            if curr["count"] > self.maxPrefixCount:
                self.maxPrefixCount = curr["count"]
                self.maxPrefixLength = i +1
                self.maxPrefixFullString = string
            elif curr["count"] == self.maxPrefixCount and i + 1 > self.maxPrefixLength:
                self.maxPrefixLength = i + 1
                self.maxPrefixFullString = string
        
