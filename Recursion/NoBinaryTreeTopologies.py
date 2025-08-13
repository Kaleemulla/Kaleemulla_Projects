def numberOfBinaryTreeTopologies(n, memo={0:1}):
    # Initiatized memo for if n==0, return 1
    if n in memo:
        return memo[n]
    numTrees = 0
    for left in range(n):
        right = n-1-left
        leftTrees = numberOfBinaryTreeTopologies(left)
        rightTrees = numberOfBinaryTreeTopologies(right)
        numTrees += leftTrees * rightTrees

    memo[n] = numTrees
    return numTrees
