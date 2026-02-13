import numpy as np
from scipy.signal import find_peaks

from ..fuzzy_utils.alpha_cuts import all_alpha_cuts
from ..fuzzy_utils.alpha_cuts import reconstruct_curve, centroid_from_curve

def mu_times_payoff(membership_value, N, Number_of_grid_points = 500, N_range = [80, 120]):
    """ This function calculates the payoff value of not tkaing action - mu * payoff.
    
        The calculations are performed using alpha cuts. 
    
    Args:
        membership_value np.ndarray( 1dim ): contains the membership values curve )
        N (callable): a function that maps payoff_grid to payoff values (the payoff function)
        Number_of_grid_points (int, optional): Number of grid points (the same size as membership value). Defaults to 500.
        N_range (list, optional): The nonzero (the core) interval of the payoff function. Defaults to [80, 120].

    Returns:
        List: [cog_taking_no_action (float) - center of gravity of mu * payoff, 
        mu_curve_taking_no_action (np.ndarray) - the reconstructed mu * payoff curve, 
        x_grid_taking_no_action (np.ndarray) - the x grid for mu * payoff curve, 
        x_of_MoM_taking_no_action (np.ndarray) - the x values of mean of maxima point for mu * payoff curve ]
    """    
    membership_grid = np.linspace(0, 1, membership_value.shape[0])
    
    payoff_grid = np.linspace(N_range[0], N_range[1], Number_of_grid_points)
    payoff_values = N(payoff_grid)
    # Find all the alpha cuts for the payoff values
    payoff_cuts = all_alpha_cuts(payoff_values, payoff_grid,  np.unique(membership_value))
    payoff_cuts[1.0] = [[100, 100]]
 
    mu_x_payoff_cuts = {}
    keys = list(payoff_cuts.keys())
    # Find all the alpha cuts for the membership function
    mu_cuts = all_alpha_cuts(membership_value, membership_grid, np.unique(membership_value))   

    # Goes trough all the alpha cuts and computes mu * payoff w.r.t. alpha cut
    for alpha in keys:
            try:
                # Has to do 1 - mu, becuase we want the better quality to be closer to 1, not 0 here
                mu_x_payoff_cuts[alpha] = [(payoff_cuts[alpha][0][0] * (1-mu_cuts[alpha][0][1])).item(), 
                                                  (payoff_cuts[alpha][0][1] * (1-mu_cuts[alpha][0][0])).item()]
            except:
                print('Skipping alpha:', alpha)
      
    cog_taking_no_action, mu_curve_taking_no_action, x_grid_taking_no_action = reconstruct_curve(mu_x_payoff_cuts)
    peak_indices, _ = find_peaks(mu_curve_taking_no_action )

    # There is an issue, whne no peaks ar found, that happens in cases
    # when the peak is at 0 or 1. Manually assign the mean of maxima then.
    if peak_indices.size == 0:
        if mu_curve_taking_no_action[0] > mu_curve_taking_no_action[-1]:
            x_of_MoM_taking_no_action = x_grid_taking_no_action[0]
        else:
            x_of_MoM_taking_no_action = x_grid_taking_no_action[-1]
    else:
        x_of_MoM_taking_no_action = x_grid_taking_no_action[peak_indices]

    return cog_taking_no_action, mu_curve_taking_no_action, x_grid_taking_no_action, x_of_MoM_taking_no_action



def taking_action_N_MuC(membership_value,  N, c, Number_of_grid_points = 500, N_range = [80, 120], c_range = [30, 60]):
    """ This function calculates the payoff value of not tkaing action - payoff - mu * cleaning cost.
    
        The calculations are performed using alpha cuts. 
    
    Args:
        membership_value np.ndarray( 1dim ): contains the membership values curve )
        N (callable): a function that maps payoff_grid to payoff values (the payoff function)
        c (callable): a function that maps cleaning_cost_grid to cleaning cost values (the cleaning cost function)
        Number_of_grid_points (int, optional): Number of grid points (the same size as membership value). Defaults to 500.
        N_range (list, optional): The nonzero (the core) interval of the payoff function. Defaults to [80, 120].
        c_range (list, optional): The nonzero (the core) interval of the cleaning cost function. Defaults to [30, 60].

    Returns:
        List: [cog_taking_action (float) - center of gravity of payoff - mu * cleaning cost, 
        mu_curve_taking_action (np.ndarray) - the reconstructed payoff - mu * cleaning cost curve, 
        x_grid_taking_action (np.ndarray) - the x grid for payoff - mu * cleaning cost curve, 
        x_of_MoM_taking_action (np.ndarray) - the x values of mean of maxima point for payoff - mu * cleaning cost curve ]
    """ 
    
    membership_grid = np.linspace(0, 1, Number_of_grid_points)
    payoff_grid = np.linspace(N_range[0], N_range[1], Number_of_grid_points)
    payoff_values = N(payoff_grid)
    # Find all the alpha cuts for the payoff values
    payoff_cuts = all_alpha_cuts(payoff_values, payoff_grid,  np.unique(membership_value))  
    payoff_cuts[1.0] = [[100, 100]]
 
    cleaning_cost_grid = np.linspace(c_range[0], c_range[1], Number_of_grid_points)
    cleaning_cost_value = c(cleaning_cost_grid)
    # Find all the alpha cuts for the cleaning cost values
    cleaning_cost_cuts = all_alpha_cuts(cleaning_cost_value, cleaning_cost_grid,  np.unique(membership_value))
    cleaning_cost_cuts[1.0] = [[45, 45]]
    payoff_minus_mu_x_c_cuts = {}
    keys = list(payoff_cuts.keys())
    mu_cuts = all_alpha_cuts(membership_value, membership_grid, np.unique(membership_value))   
    
    # Goes trough all the alpha cuts and computes mu * payoff w.r.t. alpha cut
    for alpha in keys:                             
        try:       
            payoff_minus_mu_x_c_cuts[alpha] = [(payoff_cuts[alpha][0][0] - cleaning_cost_cuts[alpha][0][1] * (mu_cuts[alpha][0][1])).item(), 
                                               (payoff_cuts[alpha][0][1] - cleaning_cost_cuts[alpha][0][0] * (mu_cuts[alpha][0][0])).item()]
        except:
            print('Skipping alpha:', alpha)
            # print(N_range, c_range)
  
            # print(cleaning_cost_cuts.keys())
            # print(c(cleaning_cost_grid))
            # print(cleaning_cost_cuts , "cleaning_cost_cut")
            
    cog_taking_action, mu_curve_taking_action, x_grid_taking_action = reconstruct_curve(payoff_minus_mu_x_c_cuts)
    peak_indices, _ = find_peaks(mu_curve_taking_action)
 
    # In case there is an issue of no peak. It happens when the 
    # maxima is reached at only 1 point at the edge of the grid
    if peak_indices.size == 0:
        if mu_curve_taking_action[0] > mu_curve_taking_action[-1]:
            x_of_MoM_taking_no_action = x_grid_taking_action[0]
        else:
            x_of_MoM_taking_no_action = x_grid_taking_action[-1]
    else:
        try:
            x_of_MoM_taking_no_action = x_grid_taking_action[peak_indices[1]]
        except IndexError:
            print(peak_indices, "peak_indices")
            x_of_MoM_taking_no_action = x_grid_taking_action[peak_indices[0]]

    return cog_taking_action, mu_curve_taking_action, x_grid_taking_action, x_of_MoM_taking_no_action

def simple_payoff_function(membership_value,  N, c, Number_of_grid_points = 500, N_range = [80, 120], c_range = [30, 60]):
    """ Simple function, that combines the two payoff function calculations:
    
            1) mu * payoff (not taking action)
            2) payoff - mu * cleaning cost (taking action)
    
        For now, it simply returns the values. 
    
    Args:
        membership_value np.ndarray( 1dim ): contains the membership values curve )
        N (callable): a function that maps payoff_grid to payoff values (the payoff function)
        c (callable): a function that maps cleaning_cost_grid to cleaning cost values (the cleaning cost function)
        Number_of_grid_points (int, optional): Number of grid points (the same size as membership value). Defaults to 500.
        N_range (list, optional): The nonzero (the core) interval of the payoff function. Defaults to [80, 120].
        c_range (list, optional): The nonzero (the core) interval of the cleaning cost function. Defaults to [30, 60].

    Returns:
        Dict: {'Taking_no_action': Taking_no_action (List) - contains the results for taking no action,
               'Taking_action': Taking_action (List) - contains the results for taking action }

    """ 
    Taking_no_action = mu_times_payoff(membership_value, N, Number_of_grid_points, N_range)
    Taking_action = taking_action_N_MuC(membership_value,  N, c, Number_of_grid_points, N_range , c_range )
    return {'Taking_no_action': Taking_no_action, 'Taking_action': Taking_action}