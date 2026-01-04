import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from ..fuzzy_utils.fuzzy_utils import modifier, R_implication  
from ..fuzzy_utils.alpha_cuts import all_alpha_cuts, create_1d_hat_function_scipy, create_1d_gaussian_function
 
from ..utils.utils import update_permutation
        
class ImplicativeInferenceSystem():
    def __init__(self, rule_grid,
                 output_membership_degree = np.array([0, 0.25, 0.5, 0.75, 1]),
                 number_of_grid_points = 500,
                 type_of_output_funs = 'Gaussian',
                 type_of_modifier = None, 
                 type_of_tnorm = 'luk'):
        
        self.rule_grid = rule_grid
        self.output_membership_degree = output_membership_degree
        self.number_of_grid_points = number_of_grid_points
        self.type_of_output_funs = type_of_output_funs
        self.modifier_type = type_of_modifier
        self.type_of_tnorm = type_of_tnorm
        self.number_of_grid_points = number_of_grid_points
        self.number_of_rules = output_membership_degree.shape[0]
        
        self.output_membership_grid = np.linspace(0, 1, self.number_of_grid_points )
        self.input_rules  = []
        self.output_rules = []
        self.define_membership_functions()
  
    def inference(self, sample):

        if self.modifier_type == 'atl' or self.modifier_type == 'atm' or self.modifier_type == None:
            final_quality_membership = self.perform_inference(sample, self.modifier_type)
        
        elif self.modifier_type == 'altm':
            membership_deg_atl = final_quality_membership = self.perform_inference(sample, 'atl')
            membership_deg_atm = final_quality_membership = self.perform_inference(sample, 'atm')
            final_quality_membership = np.minimum(membership_deg_atl, membership_deg_atm)
                 
        # Calculate the center of gravity of the resulting number
        numerator = np.trapezoid(self.output_membership_grid * final_quality_membership,
                                 self.output_membership_grid)
        
        denominator = np.trapezoid(final_quality_membership,
                                   self.output_membership_grid)
        center_of_gravity = numerator / denominator  
        
        return center_of_gravity, final_quality_membership 
            
    
    def perform_inference(self, sample, modifier_type = 'atl'):
        quality = []
        rule_permutation = [0] * 5
        # Goes trough all permutations of hat functions:    
        while True:
            min_val = 1
            # To perserve the 'One out, all out' evaluation as it was in crisp case,
            # for each rule permutation, they will fire to 'worst' (in this case with highest index)
            # index at output function. 
            rule_to_activate = max(rule_permutation)  # int(round(sum(rule_permutation)/len(rule_permutation)))
            
            # Given a permutation of hat functions, calculate the value 
            # and find minimal value out of them
            # Becasue  for some data, they could be outside of the interval, 
            # they are projected back 
            for rule_id_i, function_id in enumerate(rule_permutation):

                sample_to_take = max(self.rule_grid[rule_id_i][0], min(sample[rule_id_i], self.rule_grid[rule_id_i][-1]))
                min_val =  min(min_val,  modifier(sample_to_take,
                                            self.input_rules[rule_id_i][function_id], 
                                            set_range=np.linspace(self.rule_grid[rule_id_i][0], self.rule_grid[rule_id_i][-1], self.number_of_grid_points),
                                            type_of_tnorm=self.type_of_tnorm, modifier_type=modifier_type))
                
                # if one of rules is 0, the output will also be 0 so no reason to calculate other rule values
                if min_val == 0:
                    break
            value_to_append = R_implication(min_val, modifier(self.output_membership_grid,
                                                                self.output_rules[rule_to_activate],
                                                                self.output_membership_grid,
                                                                type_of_tnorm = self.type_of_tnorm,
                                                                modifier_type = modifier_type),
                                                type_of_implication = self.type_of_tnorm)
            quality.append(value_to_append)
            # update the permutation (uses similar principle as to how addition for binary numbers work)
            rule_permutation, to_end = update_permutation(rule_permutation, 0)
            # if all options are tried, break the loop
            if to_end:
                break
            
        # Perform maximum over all grid points, giving the final fuzzy membership number.
        resulting_rule = np.array(quality)
        final_quality_membership = np.min(resulting_rule, axis=0)
        return final_quality_membership


    def define_membership_functions(self):
        # Define output (y) membership functions
        self.output_rules = []
        for i in range(self.rule_grid.shape[0]):
            if self.type_of_output_funs == 'Gaussian':
                self.output_rules.append( create_1d_gaussian_function(self.output_membership_degree, i) )
            elif self.type_of_output_funs == 'Hat':
                self.output_rules.append(create_1d_hat_function_scipy(self.output_membership_degree, i)) 
                
        # Define input (x) membership functions
        for row in self.rule_grid:
            hat_funs = []
            for i in range(row.shape[0]):
                hat_funs.append(create_1d_hat_function_scipy(row, i))
            self.input_rules.append(hat_funs)
            
if __name__ == "__main__":
    row = np.arange(1, 6)

    # Tile the row 5 times vertically

    matrix = np.tile(row, (5, 1))
    print(matrix)
    
    dummy_system = ImplicativeInferenceSystem(matrix, type_of_modifier='altm')
    center_of_gravity, final_quality_membership, expected_mu_value = dummy_system.inference(np.array([1, 1, 1, 1, 1]))
    print(center_of_gravity)
    print(expected_mu_value)
    
    plt.plot(final_quality_membership)
    plt.show()
    
    