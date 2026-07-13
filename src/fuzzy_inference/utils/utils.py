import numpy as np


def update_permutation(permutations, i, max_value=5):
    """Function to move to the next permutation of rules. 

    Args:
        permutations (List): what permutation of rules is to active
        i (id): to which element +1 should be added
        max_value (int): the maximum value for each element in the permutation

    Returns:
        permutations (List):  returns list of active rules for next  iteration
        flag (Bool): if True, all combinations have been ran. 
    """    
    flag = False
    if i == len(permutations):
        return permutations, True
    permutations[i] += 1
    
    if permutations[i] == max_value:
        permutations[i] = 0
        permutations, flag = update_permutation(permutations, i + 1)
        
    return permutations, flag