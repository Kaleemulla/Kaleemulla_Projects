class LinkedList:
    def __init__(self, value):
        self.value = value
        self.next = None


def reverseLinkedList(head):
    p1 = None
    p2 = head

    p3 = p2.next if p2 else None

    while p2:
        #p3 = p2.next
        p2.next = p1

        p1 = p2
        p2 = p3
        p3 = p3.next if p3 else None

    return p1
