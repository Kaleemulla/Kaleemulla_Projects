def levenshteinDistance(str1, str2):
    dp = [[x for x in range(len(str2)+1)] for _ in range(len(str1)+1)] # Initialize first row with 0,1,2,3,4..

    for i in range(len(str1)+1): # Initialize first column with 0,1,2,3,4...
        dp[i][0] = i

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] # No. of substitution when both are same is 0
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) # when diff then 1 + min sub

    return dp[-1][-1]
    pass
