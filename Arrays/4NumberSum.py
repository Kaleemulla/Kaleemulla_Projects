def fourNumberSum(array, targetSum):
    allpairs = {}
    result = []

    # Similar to 2 number, instead of adding single value to hash and then check if diff was in hash
    # Add a pair of number to hash with key as sum, add when current pointer i is second number of pair
    
    for i in range(1, len(array)-1): # Skip first,last because it wont add anything to hash since left of 0 is null and left of last is already in hash
        for j in range(i+1, len(array)): # for first pair in hash, find second pair with i, i+1
            sum = array[i] + array[j]
            currentdiff = targetSum - sum

            # If currentdiff in hash, there is a pair with target = [existingpair] + [currentpair]
            if currentdiff in allpairs:
                for pair in allpairs[currentdiff]:
                    result.append(pair + [array[i], array[j]])
                    
            # if currentdiff not in hash, skip and do nothing until i is at second item of pair

        # Add all pairs in left of i to hash with sum as its value
        for k in range(0, i): # We are adding pairs to hash only when its second item is on current pointer. This is to avoid revisiting same numbers. [1,2], add to hash when cp is 2
            previoussum = array[i] + array[k]
            if previoussum not in allpairs:
                allpairs[previoussum] = [[array[k], array[i]]] # Adding k first and i=cp second
            else:
                allpairs[previoussum].append([array[k], array[i]])
                
    return result
    pass
