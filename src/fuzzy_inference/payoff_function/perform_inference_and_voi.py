import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

 
from fuzzy_inference.rule_bases.Implicative_model import ImplicativeInferenceSystem
from fuzzy_inference.fuzzy_utils.fuzzy_sets   import FuzzyNumber
from fuzzy_inference.payoff_function.perform_price_analysis import full_voi_analysis, plot_payoff_plots


def perform_infernece_and_evaluation_with_data(data_sample, antecendant_qualityGrid, step_size = 20, Number_of_grid_points = 100,
                                               type_of_tnorm = 'luk', sample_name = 'dummy_sample', use_the_same_tnorm = False, 
                                               cost_value_config = None, payoff_function= None):
    
    """ This function pperforms: 
            1. inference on the passed data and plots membership degree
            2. Performs Voi analysis of given data/function. Otherwise uses a basic
            payoff function. 
    
    The sample data is in order: BSP5 N_NH4+ N_total O_2 P_total
    
    Args:
        data_sample (np.ndarray): Sample data for inference.
        antecendant_qualityGrid (np.ndarray): Antecedent quality grid for inference.
        step_size (int): Step size for parameter ranges. 
        Number_of_grid_points (int): Number of grid points for fuzzy sets. 
        type_of_tnorm (str): Type of t-norm for inference. Defaults to Lukasiewicz ('luk').
        sample_name (str): Name of the sample for labeling plots. 
        use_the_same_tnorm (bool): Whether to use the same t-norm for aggregation.
        cost_value_config (dict): Configuration for cost values. If None, a default configuration is used.
        payoff_function (callable): Payoff function to use. If None, a default payoff function is defined.
    Returns:
        tuple: A tuple containing the results matrix, parameter ranges, and membership value.
    """    
    # First perform inference
    type_1_system = ImplicativeInferenceSystem(antecendant_qualityGrid, type_of_modifier='altm',
                                               number_of_grid_points=Number_of_grid_points,
                                               type_of_tnorm=type_of_tnorm, aggregate_with_same_tnorm=use_the_same_tnorm)
    membership_value  = type_1_system.inference(data_sample)
 

    print(f"The membership function of sample {sample_name}: ")
    membership_value.plot(label = f'{sample_name}')

    print(" ----------------- Full price analists -----------------")
    
    if cost_value_config is None and payoff_function is None:
        cost_value_config = {
            'N': {
                'bounds': (1, 200),
                'left_spread': 20,    
                'right_spread': 20    
            },
            'C': {
                'bounds': (1, 150),
                'left_spread': 25,    
                'right_spread': 25 
            }
        }

        # Define payofff function
        def payoff_function(u, mu, N, C):
            return (N + (-1) * mu * C) * u  +  ( N * -1 * (mu - 1)) * (1 - u)
        
    elif cost_value_config is None or payoff_function is  None:
        raise ValueError("Both cost_value_config and payoff_function must be provided together.")
        
    # Run the system
    results_matrix, ranges = full_voi_analysis(
        membership_value=membership_value,
        step_size=step_size,
        param_config=cost_value_config,
        payoff_formula=payoff_function,
        u_space=np.linspace(0, 1, 100)
    )
    return results_matrix, ranges, membership_value
     