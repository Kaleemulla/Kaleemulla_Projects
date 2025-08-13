def largestRange(array):
    hash = {}
    largest = 0
    result = []
    
    for num in array:
        hash[num] = True # Put all in hash for access time O(1)

    for num in array:
        if not hash[num]:
            continue

        hash[num] = False # Make it false so that they are not revisited again and continued with above
        
        i, j, length = num-1, num + 1, 1
        while i in hash: # For each num, check how much to its left range does num exist 
            hash[i] = False # Make it false so that they are not revisited again and continued
            i -= 1
            length += 1

        while j in hash: # same find range to right
            hash[j] = False
            j += 1
            length += 1

        if length > largest:
            largest = length
            result = [i+1, j-1]

    return result
    pass
