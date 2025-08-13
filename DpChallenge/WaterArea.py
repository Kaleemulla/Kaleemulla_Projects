def waterArea(heights):
    max_right = [h for h in heights]
    max_left = [h for h in heights]
    
    for i in range(1, len(heights)):
        max_left[i] = max(max_left[i-1], max_left[i]) # Max of left array

    for i in reversed(range(0, len(heights)-1)):
        max_right[i] = max(max_right[i+1], max_right[i]) # Max of right array

    min_max = [min(max_right[i], max_left[i]) - heights[i] for i in range(len(heights))] # Reduce building size from min of left and right array
    sums = sum(i for i in min_max) # Sum of all waters above building

    return sums
    
    pass
