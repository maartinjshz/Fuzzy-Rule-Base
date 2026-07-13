import numpy as np

def T_norm(a,b, type_of_tnorm = "luk"):
    if type_of_tnorm == 'min':
        return np.minimum(a,b)
    elif type_of_tnorm == 'prod':
        return a * b
    elif type_of_tnorm == 'luk':
        return np.maximum(0, a + b - 1)
    
def R_implication(a,b, type_of_implication = "luk"):
    if type_of_implication == "min":
        condition = (a <= b)
        return np.where(condition, 1.0, b)
        
    elif type_of_implication == "prod":
        condition = (a <= b)
        result_b_over_a = np.divide(b, a, out=np.zeros_like(b), where=(a!=0))
        return np.where(condition, 1.0, result_b_over_a)
        
    elif type_of_implication ==  "luk":
        return np.minimum(1, 1 - a + b )
 
def T_E_inequality(a, b, type_of_tnorm = "luk"):
    if type_of_tnorm == "prod":
        return np.where(a <= b, 1, np.exp( - np.abs(a-b)))
    
    elif type_of_tnorm == "luk":
        result = np.clip(1 - np.abs(a - b), 0, None)
        return np.where(a <= b, 1.0, result)
    
    
def vectorized_t_norm(X, type_of_tnorm = "prod"):
    if type_of_tnorm == 'prod':
        # Calculates the product of all elements in each column.
        return np.prod(X, axis=0)

    # --- Gödel (Minimum) T-norm ---
    elif type_of_tnorm == 'min':
        # Finds the minimum value in each column.
        return np.min(X, axis=0)

    # --- Łukasiewicz T-norm ---
    elif type_of_tnorm == 'luk':
        # Uses the efficient n-ary formula: max(0, sum(x_i) - (n-1))
        num_rows = X.shape[0]
        return np.maximum(0, np.sum(X, axis=0) - (num_rows - 1))
    
    
def modifier(x, fuzzy_set, set_range,  modifier_type = 'atl', type_of_tnorm = 'luk'):
    # Ensure x is a NumPy array for consistent broadcasting
    x = np.asarray(x)
    
    # Evaluate the fuzzy set over its range
    fuzzy_values = fuzzy_set(set_range)

    # Use broadcasting to compare each element of x with the entire set_range
    # This creates a 2D boolean mask.
    if modifier_type == 'atl':
        mask = set_range[:, np.newaxis] <= x
    elif modifier_type == 'atm':
        mask = set_range[:, np.newaxis] >= x
    elif modifier_type == None:
        return fuzzy_set(x)
        
    # Apply the mask to the fuzzy values, setting non-matching elements to 0
    # The result is a 2D array where each column corresponds to an element in x
    masked_values = np.where(mask, fuzzy_values[:, np.newaxis], 0)
    
    # Find the maximum value in each column (axis=0)
    # This returns a 1D array with the result for each element of x
    return np.max(masked_values, axis=0)



def Mean_of_Maximum_defuziffication(fuzzy_number):
    """Function for defuzzification using the Mean of Maximum method.

    Args:
        fuzzy_number (FuzzyNumber): Fuzzy number to be defuzzified
    Returns:
        float: The defuzzified value of the fuzzy number using the Mean of Maximum method
    """    
    
    if not hasattr(fuzzy_number, 'alpha_cuts'):
        raise TypeError("Input must have an 'alpha_cuts' attribute (should be a FuzzyNumber instance).")
    
    highest_alpha_level = max(fuzzy_number.alpha_cuts.keys())
    return np.mean(fuzzy_number.alpha_cuts[highest_alpha_level])



def Center_of_Gravity_defuziffication(fuzzy_number):
    """Function for defuzzification using the Center of Gravity method.

    Args:
        fuzzy_number (FuzzyNumber): Fuzzy number to be defuzzified

    Returns:
        float: The defuzzified value of the fuzzy number using the Center of Gravity method
    """    
    if not hasattr(fuzzy_number, 'space_x') or not hasattr(fuzzy_number, 'membership_values'):
        raise TypeError("Input must have 'space_x' and 'membership_values' attributes (should be a FuzzyNumber instance).")
    
    # Calculate the center of gravity of the resulting number
    numerator = np.trapezoid(fuzzy_number.space_x * fuzzy_number.membership_values,
                                fuzzy_number.space_x)
    
    denominator = np.trapezoid(fuzzy_number.membership_values,
                                fuzzy_number.space_x)
    center_of_gravity = numerator / denominator  
    
    return center_of_gravity