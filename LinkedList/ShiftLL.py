class LinkedList:
    def __init__(self, value):
        self.value = value
        self.next = None


def shiftLinkedList(head, k):
    tail = head
    leng = 1
    
    while tail.next:
        leng += 1
        tail = tail.next

    offset = abs(k) % leng

    if offset == 0:
        return head
        
    pos = leng - offset if k > 0 else offset
    curr = head
    
    for i in range(1, pos):
        curr = curr.next

    newHead = curr.next
    curr.next = None
    tail.next = head

    return newHead

    
        
