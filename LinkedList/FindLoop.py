class LinkedList:
    def __init__(self, value):
        self.value = value
        self.next = None


def findLoop(head):
    slow = head.next
    fast = head.next.next # Because else while next will immediately be met :)

    while slow != fast:
        slow = slow.next
        fast = fast.next.next

    slow = head

    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow
