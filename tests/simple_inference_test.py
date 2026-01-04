import pytest
import numpy as np
import pandas as pd
import os

from fuzzy_inference.fuzzy_utils.alpha_cuts import create_1d_hat_function_scipy 
from fuzzy_inference.payoff_function.simple_payoff_functions import simple_payoff_function
from fuzzy_inference.rule_bases.Implicative_model import ImplicativeInferenceSystem

def test_data_integrity():
    file_path = '../conference_material/Data/River_quality_data_2023.csv'
    
    # 1. Ensure file exists where expected
    assert os.path.exists(file_path), f"Data file {file_path} not found!"
    
    # 2. Ensure columns haven't been renamed
    df = pd.read_csv(file_path)
    required_cols = ['Station_type', 'BSP5', 'N_NH4+', 'N_total', 'O_2', 'P_total']
    for col in required_cols:
        assert col in df.columns, f"Column {col} is missing from the dataset!"
        
    # 3. Ensure no empty rows were added by accident
    assert not df.isnull().values.any(), "The dataset contains empty (NaN) values!"

def simple_inference_test():
    row = np.arange(1, 6)
    matrix = np.tile(row, (5, 1))   
    dummy_system_1 = ImplicativeInferenceSystem(matrix, type_of_modifier='altm')
    dummy_system_2 = ImplicativeInferenceSystem(matrix, type_of_modifier='atm', type_of_tnorm='prod')
    dummy_system_3 = ImplicativeInferenceSystem(matrix, type_of_modifier='atl', type_of_tnorm='min')
    
    center_of_gravity_1, final_quality_membership_1 = dummy_system_1.inference(np.array([1, 2, 1, 3, 1]))
    center_of_gravity_2, final_quality_membership_2 = dummy_system_2.inference(np.array([1, 2, 1, 3, 1]))
    center_of_gravity_3, final_quality_membership_3 = dummy_system_3.inference(np.array([1, 2, 1, 3, 1]))

    assert isinstance(center_of_gravity_1, float), "Center of gravity should be a float."
    assert isinstance(center_of_gravity_2, float), "Center of gravity should be a float."
    assert isinstance(center_of_gravity_3, float), "Center of gravity should be a float."
    
    assert np.round(center_of_gravity_1, 3) == 0.5
    assert final_quality_membership_1.shape == (dummy_system_1.number_of_grid_points,), "Final quality membership shape mismatch."
    



def test_simple_payoff_function():
    # Perform the simple inference 
    row = np.arange(1, 6)
    matrix = np.tile(row, (5, 1))   
    dummy_system = ImplicativeInferenceSystem(matrix, type_of_modifier='altm')
    center_of_gravity, final_quality_membership  = dummy_system.inference(np.array([1, 2, 1, 3, 1]))
    
    # Defone the membership functions for payoff and cleaning costs
    N = create_1d_hat_function_scipy(np.array([80, 100, 120]), 1)
    C = create_1d_hat_function_scipy(np.array([30, 45 , 60]), 1)
        
    # Calculate the payff values for both taking and not taking the action
    calculate_the_payoff_values = simple_payoff_function(final_quality_membership,  N, C)
    # Take the both payoff values with deffuzified values
    cog_taking_action_NxMu, mu_curve_taking_NxMu, x_grid_taking_NxMu, x_of_maxima_NxMu = calculate_the_payoff_values['Taking_no_action']
    cog_taking_action_N_MuC, mu_curve_taking_N_MuC, x_grid_taking_N_MuC, x_of_maxima_N_MuC = calculate_the_payoff_values['Taking_action']
    
    # Do a sanity scheck of the results
    assert 'Taking_no_action' in calculate_the_payoff_values
    assert 'Taking_action' in calculate_the_payoff_values
    
    assert np.isfinite(cog_taking_action_NxMu)
    assert np.isfinite(cog_taking_action_N_MuC)
    
    assert np.all(mu_curve_taking_NxMu >= 0) and np.all(mu_curve_taking_NxMu <= 1)
    assert np.all(mu_curve_taking_N_MuC >= 0) and np.all(mu_curve_taking_N_MuC <= 1)
    
    assert len(x_grid_taking_NxMu) == len(mu_curve_taking_NxMu)
    assert len(x_grid_taking_N_MuC) == len(mu_curve_taking_N_MuC)
    
    # Check values of center of gravity
    assert cog_taking_action_NxMu == pytest.approx(53.33, rel=1e-3)
    assert cog_taking_action_N_MuC == pytest.approx(74.97, rel=1e-3)
    
 
    
 