class LinkedList:
    def __init__(self, value):
        self.value = value
        self.next = None


def mergeLinkedLists(headOne, headTwo):
    p1 = headOne
    prev = None
    p2 = headTwo

    while p1 and p2:
        if p1.value < p2.value: # Keep moving in the list-1
            prev = p1 # Track prev in p1
            p1 = p1.next
        else:
            if prev:
                prev.next = p2 # Make prev track list2
                
            prev = p2
            p2 = p2.next
            prev.next = p1

    if not p1:
        prev.next = p2 # list1 exhausted, point all in lis2 to prev.next

    return headOne if headOne.value < headTwo.value else headTwo
        
