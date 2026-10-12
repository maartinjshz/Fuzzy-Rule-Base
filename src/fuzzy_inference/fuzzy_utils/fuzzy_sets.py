import numpy as np
import matplotlib.pyplot as plt

from fuzzy_inference.fuzzy_utils.alpha_cuts import all_alpha_cuts
from ..fuzzy_utils.fuzzy_utils import Mean_of_Maximum_defuziffication, Center_of_Gravity_defuziffication
from ..fuzzy_utils.alpha_cuts import reconstruct_curve_from_alpha_cuts

class FuzzyNumber:
    def __init__(self, 
                 membership_values = np.linspace(0, 1, 101), membership_space = None,
                 space_x = np.linspace(0, 1, 101), lower = None, upper = None):

        self.membership_values = membership_values
        self.membership_space = membership_space
        self.space_x = space_x
        
        if self.membership_space is None:
            self.membership_space = np.unique(self.membership_values)
        
        if lower is None and upper is None:
            lower, upper = all_alpha_cuts(self.membership_values,
                                            self.space_x,
                                            self.membership_space)
        elif membership_values is None or space_x is None:
            raise ValueError("If alpha_cuts is not provided, both membership_values and space_x must be provided.")
        
        
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
                
        # if alpha_cuts is not None:
        #     self.alpha_cuts = alpha_cuts
        #     if self.alpha_cuts.keys() != set(membership_space):
        #         raise ValueError("Alpha cuts must be provided for all levels in the membership space.")
        # else:
        #     # if membership_values is None or space_x is None:
        #     #     raise ValueError("If alpha_cuts is not provided, both membership_values and space_x must be provided.")
        #     self.alpha_cuts = all_alpha_cuts(self.membership_values,
        #                                      self.space_x,
        #                                      np.unique(self.membership_space))
            
        # # if self.space_x is None:
        # #     self.space_x = np.linspace(self.alpha_cuts[0.0][0], self.alpha_cuts[0.0][1], len(self.membership_space))

        # if self.membership_values is None:
        #     self.membership_values = reconstruct_curve_from_alpha_cuts(self.alpha_cuts, self.space_x)
        # Define such properties as Mean of Maxima and Center of Gravity for the fuzzy number
        self.mean_of_maxima = Mean_of_Maximum_defuziffication(self)
        self.center_of_gravity = Center_of_Gravity_defuziffication(self)

    @classmethod
    def triangular(cls, a, b, c, membership_space = np.linspace(0, 1, 101)):
        """Factory method to create a Triangular Fuzzy Number (a, b, c)"""
        
        space_x = np.linspace(a, c, membership_space.size)
        membership_values = np.zeros_like(space_x)
        for i, x in enumerate(space_x):
            if a <= x <= b:
                membership_values[i] = (x - a) / (b - a)
            elif b < x <= c:
                membership_values[i] = (c - x) / (c - b)
                
        membership_value_grid = np.linspace(0, 1, membership_space.size)
        lower, upper = all_alpha_cuts(membership_values, space_x, membership_value_grid)
        space_x_new =  np.sort(np.concatenate([lower, upper ]))
        return cls(membership_values=np.concatenate([membership_value_grid, membership_value_grid[::-1]]),
                   membership_space= membership_value_grid,
                   space_x=space_x_new, lower=lower, upper=upper)

    @classmethod
    def gaussian(cls, a, b, middle, sigma,  membership_space = np.linspace(0, 1, 101)):
        """Factory method to create a Gaussian Fuzzy Number (a, b, middle, sigma)"""
        space_x = np.linspace(a,b, membership_space.size+1)
        gaussian_fun_values = np.exp(-  ((space_x - middle) / sigma) ** 2)

        return cls(membership_values=gaussian_fun_values,
                   membership_space=np.sort(np.unique(gaussian_fun_values)),
                   space_x=space_x)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            cand = np.stack([
                self.lower + other,
                self.upper + other,
            ], axis=0)
    
            new_lower, new_upper = np.min(cand, axis=0), np.max(cand, axis=0)
            space_x_new =  np.sort(np.concatenate([new_lower, new_upper]))
            
            return FuzzyNumber(self.membership_values,
                        self.membership_space,
                        space_x_new, new_lower, new_upper)
            
        elif not isinstance(other, FuzzyNumber):
            raise TypeError("Operations can only be performed between two FuzzyNumbers.")
        elif not np.array_equal(self.membership_space, other.membership_space):
            raise ValueError("FuzzyNumbers must have the same membership space for operations.")
        
        cand = np.stack([
            self.lower + other.lower,
            self.lower + other.upper,
            self.upper + other.lower,
            self.upper + other.upper,
        ], axis=0)
        
        new_lower, new_upper = np.min(cand, axis=0), np.max(cand, axis=0)
        space_x_new =  np.sort(np.concatenate([new_lower, new_upper]))
        return FuzzyNumber(self.membership_values,
                 self.membership_space,
                 space_x_new, new_lower, new_upper)
        
    def __radd__(self, other):
        """Handles right-side addition (e.g., 5 + FuzzyNumber)"""
        return self.__add__(other)
       
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            cand = np.stack([
                self.lower - other,
                self.upper - other,
            ], axis=0)
    
            new_lower, new_upper = np.min(cand, axis=0), np.max(cand, axis=0)
            space_x_new =  np.sort(np.concatenate([new_lower, new_upper]))
            
            return FuzzyNumber(self.membership_values,
                        self.membership_space,
                        space_x_new, new_lower, new_upper)
            
        elif not isinstance(other, FuzzyNumber):
            raise TypeError("Operations can only be performed between two FuzzyNumbers.")
        elif not np.array_equal(self.membership_space, other.membership_space):
            raise ValueError("FuzzyNumbers must have the same membership space for operations.")
        
        cand = np.stack([
            self.lower - other.lower,
            self.lower - other.upper,
            self.upper - other.lower,
            self.upper - other.upper,
        ], axis=0)
        
        new_lower, new_upper = np.min(cand, axis=0), np.max(cand, axis=0)
        space_x_new =  np.sort(np.concatenate([new_lower, new_upper]))
        return FuzzyNumber(self.membership_values,
                 self.membership_space,
                 space_x_new, new_lower, new_upper)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            cand = np.stack([
                self.lower * other,
                self.upper * other,
            ], axis=0)
    
            new_lower, new_upper = np.min(cand, axis=0), np.max(cand, axis=0)
            space_x_new =  np.sort(np.concatenate([new_lower, new_upper]))
            
            return FuzzyNumber(self.membership_values,
                        self.membership_space,
                        space_x_new, new_lower, new_upper)
            
        elif not isinstance(other, FuzzyNumber):
            raise TypeError("Operations can only be performed between two FuzzyNumbers.")
        elif not np.array_equal(self.membership_space, other.membership_space):
            raise ValueError("FuzzyNumbers must have the same membership space for operations.")
        
        cand = np.stack([
                    self.lower * other.lower,
                    self.lower * other.upper,
                    self.upper * other.lower,
                    self.upper * other.upper,
                ], axis=0)
                
        new_lower, new_upper = np.min(cand, axis=0), np.max(cand, axis=0)
        space_x_new =  np.sort(np.concatenate([new_lower, new_upper]))
        
        return FuzzyNumber(self.membership_values,
                    self.membership_space,
                    space_x_new, new_lower, new_upper)

    def __rmul__(self, other):
        """Handles right-side multiplication (e.g., 5 * FuzzyNumber)"""
        return self.__mul__(other)

    def plot(self, label=None, path = "",
             to_save = False, interval_colors = [
                                '#FF0000',  # Bad bad
                                '#FFA500',  # Poor
                                '#FFFF00',  # Moderate
                                '#90EE90',  # Good
                                '#00BFFF',  # High  
                            ],
             grid = np.array([0, 0.25, 0.5, 0.75, 1]),
             to_plot_intervals = True):
       
        """Quick utility to plot the membership function"""
        # The interval plotting is made
        # due to main purpose was water quality analysis.
        # This could help visualize the fuzzy number and
        # in which interval it falls inside.
        if to_plot_intervals:
            for i in range(len(grid) - 1):
                
                    if i == 0:
                        plt.hlines(y=-0.007, xmin=grid[i], xmax=(grid[i]+grid[i+1])/2, colors=interval_colors[i], linewidth=5) 
                    else:
                        plt.hlines(y=-0.007, xmin=(grid[i-1]+grid[i])/2, xmax=(grid[i]+grid[i+1])/2, colors=interval_colors[i], linewidth=5)
                
            plt.hlines(y=-0.007, xmin=(grid[-1]+grid[-2])/2, xmax=grid[-1], colors=interval_colors[-1], linewidth=5)
            
        plt.plot(self.space_x, self.membership_values, label="Membership function")
        plt.title(f"{label}" if label else "Membership function of fuzzy number", fontsize=16)
        plt.axvline(self.center_of_gravity, color='blue', linestyle='--', label=f'CoG: {self.center_of_gravity:.3f}')
        plt.axvline(self.mean_of_maxima, color='green', linestyle='--', label=f'MoM {self.mean_of_maxima:.3f}')
        plt.legend(fontsize=14,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.1))
        plt.xlabel("Quality", fontsize=16)
        plt.ylabel(r"$\alpha$ (Membership)", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
         
        
        if to_save:
            plt.savefig(
                f"{path}.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()
            
        plt.show()
    # def __repr__(self):
    #     support = self.alpha_cuts.get(0.0, ("?", "?"))
    #     core = self.alpha_cuts.get(1.0, ("?", "?"))
    #     return f"FuzzyNumber(Support=[{support[0]:.2f}, {support[1]:.2f}], Core=[{core[0]:.2f}, {core[1]:.2f}])"