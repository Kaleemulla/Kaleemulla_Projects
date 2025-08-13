def maximumSumSubmatrix(matrix, size):
    # create a matrix of same size which holds sums of all numbers from 0, i and 0, j at each position
    # a[i][j] + dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1]
    # here we do minus at last because dp[i-1][j] was sum value including dp[i-1][j-1] and same with dp[i][j-1]
    # so need to remove this once hence minus with final value

    dp = matrix[:]

    for i in range(1, len(matrix[0])):
        dp[0][i] += dp[0][i-1]

    for i in range(1, len(matrix)):
        dp[i][0] += dp[i-1][0]

    for i in range(1, len(dp)):
        for j in range(1, len(dp[0])):
            dp[i][j] += dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1]

    maxSum = float('-inf')
    
    for i in range(size-1, len(dp)): # Because min size is going to be size-1, size-1
        for j in range(size-1, len(dp[0])):
            currentSum = dp[i][j]

            touchesTopBorder = i - size < 0
            if not touchesTopBorder:
                currentSum -= dp[i-size][j]

            touchesLeftBorder = j - size < 0
            if not touchesLeftBorder:
                currentSum -= dp[i][j-size]

            touchesTopOrLeft = touchesTopBorder or touchesLeftBorder
            if not touchesTopOrLeft: # Touches both top and left border, add back the one substracted twice
                currentSum += dp[i-size][j-size]
                
            maxSum = max(maxSum, currentSum)
    
    return maxSum
