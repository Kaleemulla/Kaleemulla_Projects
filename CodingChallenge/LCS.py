def longestCommonSubsequence(str1, str2):
    # Write your code here.
    dp = [[0 for _ in range(len(str2)+1)] for _ in range(len(str1)+1)]

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = 1+dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    result = []
    i, j = len(str1), len(str2)
    
    while i>0 and j>0:
        if str1[i-1] == str2[j-1]:
            result.append(str1[i-1])
            i -=1
            j -=1
        else:
            if dp[i][j-1] > dp[i-1][j]:
                j -=1
            else:
                i -=1
                
    return result[::-1]
    pass
