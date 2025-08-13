def numberOfWaysToMakeChange(n, denoms):
    dp = [0 for _ in range(n+1)]
    dp[0] = 1

    for i in range(1, len(denoms)+1):
        for j in range(1, n+1):
            if j >= denoms[i-1]:
                dp[j] = dp[j] + dp[j-denoms[i-1]]
    return dp[-1]
