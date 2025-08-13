#T=O(v+e), S=O(v)
def twoColorable(edges):
    colors = [None for _ in edges]
    colors[0] = True # Color first item
    stack = [0] # Index of first item

    while stack:
        node = stack.pop()
        for connection in edges[node]:
            if colors[connection] is None:
                colors[connection] = not colors[node]
                stack.append(connection)
            elif colors[connection] == colors[node]:
                return False

    return True
