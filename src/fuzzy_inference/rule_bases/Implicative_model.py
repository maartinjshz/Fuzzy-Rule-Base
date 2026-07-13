import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from ..fuzzy_utils.fuzzy_utils import modifier, R_implication, vectorized_t_norm
from ..fuzzy_utils.alpha_cuts import all_alpha_cuts, create_1d_hat_function_scipy, create_1d_gaussian_function
from ..fuzzy_utils.fuzzy_sets   import FuzzyNumber
from ..utils.utils import update_permutation

from joblib import Parallel, delayed

        
class ImplicativeInferenceSystem():
    def __init__(self, rule_grid,
                 output_membership_degree = np.array([0, 0.25, 0.5, 0.75, 1]),
                 number_of_grid_points = 500,
                 type_of_output_funs = 'Gaussian',
                 type_of_modifier = None, 
                 type_of_tnorm = 'luk',
                 aggregate_with_same_tnorm = False):
        
        self.rule_grid = rule_grid
        self.output_membership_degree = output_membership_degree
        self.number_of_grid_points = number_of_grid_points
        self.type_of_output_funs = type_of_output_funs
        self.modifier_type = type_of_modifier
        self.type_of_tnorm = type_of_tnorm
        self.number_of_grid_points = number_of_grid_points
        self.number_of_rules = output_membership_degree.shape[0]
        self.aggregate_with_same_tnorm = aggregate_with_same_tnorm
        
        self.rule_ranges = [np.linspace(row[0], row[-1], self.number_of_grid_points) 
                        for row in self.rule_grid]
        
        self.output_membership_grid = np.linspace(output_membership_degree[0],
                                                  output_membership_degree[-1],
                                                  self.number_of_grid_points )
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
                 
        final_membership_fuzzy_numb = FuzzyNumber(membership_values=final_quality_membership,
                                                 membership_space=np.linspace(0, 1, self.number_of_grid_points ),
                                                 space_x=self.output_membership_grid)
        return final_membership_fuzzy_numb
            
    
    def perform_inference(self, sample, modifier_type='atl'):
        """Parallelized inference with joblib."""
        permutations = []
        rule_permutation = [0] * 5
        
        # Generate all permutations
        while True:
            permutations.append(rule_permutation.copy())
            rule_permutation, to_end = update_permutation(rule_permutation, 0)
            if to_end:
                break
        
        # Process all permutations in parallel
        quality = Parallel(n_jobs=-1)(
            delayed(self._process_permutation)(perm, sample, modifier_type) 
            for perm in permutations
        )
        
        # Perform aggregation
        resulting_rule = np.array(quality)
        if self.aggregate_with_same_tnorm:
            final_quality_membership = vectorized_t_norm(resulting_rule, type_of_tnorm=self.type_of_tnorm)
        else:
            final_quality_membership = np.min(resulting_rule, axis=0)
        
        return final_quality_membership
        
    
    def _process_permutation(self, rule_permutation, sample, modifier_type):
        """Process a single rule permutation."""
        min_val = 1
        rule_to_activate = max(rule_permutation)
        
        for rule_id_i, function_id in enumerate(rule_permutation):
            sample_to_take = max(self.rule_grid[rule_id_i][0], 
                                min(sample[rule_id_i], self.rule_grid[rule_id_i][-1]))
            min_val = min(min_val, modifier(sample_to_take,
                                            self.input_rules[rule_id_i][function_id],
                                            set_range=self.rule_ranges[rule_id_i],
                                            type_of_tnorm=self.type_of_tnorm, 
                                            modifier_type=modifier_type))
            if min_val == 0:
                break
        
        value_to_append = R_implication(min_val, 
                                    modifier(self.output_membership_grid,
                                            self.output_rules[rule_to_activate],
                                            self.output_membership_grid,
                                            type_of_tnorm=self.type_of_tnorm,
                                            modifier_type=modifier_type),
                                    type_of_implication=self.type_of_tnorm)
        return value_to_append


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
    
    