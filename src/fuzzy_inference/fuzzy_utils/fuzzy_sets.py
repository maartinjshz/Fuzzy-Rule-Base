import numpy as np
import matplotlib.pyplot as plt

from fuzzy_inference.fuzzy_utils.alpha_cuts import all_alpha_cuts
from ..fuzzy_utils.fuzzy_utils import Mean_of_Maximum_defuziffication, Center_of_Gravity_defuziffication
from ..fuzzy_utils.alpha_cuts import reconstruct_curve_from_alpha_cuts

class FuzzyNumber:
    def __init__(self, membership_values = None,
                 membership_space = np.linspace(0, 1, 101),
                 space_x = np.linspace(0, 1, 101), alpha_cuts=None):

        self.membership_values = membership_values
        self.membership_space = membership_space
        self.space_x = space_x
        
        if alpha_cuts is not None:
            self.alpha_cuts = alpha_cuts
            if self.alpha_cuts.keys() != set(membership_space):
                raise ValueError("Alpha cuts must be provided for all levels in the membership space.")
        else:
            if membership_values is None or space_x is None:
                raise ValueError("If alpha_cuts is not provided, both membership_values and space_x must be provided.")
            self.alpha_cuts = all_alpha_cuts(self.membership_values,
                                             self.space_x,
                                             np.unique(self.membership_space))
        if self.space_x is None:
            self.space_x = np.linspace(self.alpha_cuts[0.0][0], self.alpha_cuts[0.0][1], len(self.membership_space))
     
        if self.membership_values is None:
            self.membership_values = reconstruct_curve_from_alpha_cuts(self.alpha_cuts, self.space_x)
        # Define such properties as Mean of Maxima and Center of Gravity for the fuzzy number
        self.mean_of_maxima = Mean_of_Maximum_defuziffication(self)
        self.center_of_gravity = Center_of_Gravity_defuziffication(self)

    @classmethod
    def triangular(cls, a, b, c, membership_space = np.linspace(0, 1, 101)):
        """Factory method to create a Triangular Fuzzy Number (a, b, c)"""
        
        space_x = np.linspace(a, c, membership_space.size + 1)
        membership_values = np.zeros_like(space_x)
        for i, x in enumerate(space_x):
            if a <= x <= b:
                membership_values[i] = (x - a) / (b - a)
            elif b < x <= c:
                membership_values[i] = (c - x) / (c - b)
        return cls(membership_values=membership_values,
                   membership_space=membership_space,
                   space_x=space_x)

    @classmethod
    def gaussian(cls, a, b, middle, sigma,  membership_space = np.linspace(0, 1, 101)):
        """Factory method to create a Gaussian Fuzzy Number (a, b, middle, sigma)"""
        space_x = np.linspace(a,b, membership_space.size+1)
        gaussian_fun_values = np.exp(-  ((space_x - middle) / sigma) ** 2)

        return cls(membership_values=gaussian_fun_values,
                   membership_space=membership_space,
                   space_x=space_x)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            new_cuts = {}
            for alpha in self.alpha_cuts.keys():
                l, r = self.alpha_cuts[alpha]
                new_cuts[alpha] = np.amin([l + other, r + other]), np.amax([l + other, r + other])
    
            return FuzzyNumber(None,
                 self.membership_space,
                 None, new_cuts)
        elif not isinstance(other, FuzzyNumber):
            raise TypeError("Operations can only be performed between two FuzzyNumbers.")
        elif not np.array_equal(self.membership_space, other.membership_space):
            raise ValueError("FuzzyNumbers must have the same membership space for operations.")
        
        new_cuts = {}
        # Perform interval addition for each common alpha level
        for alpha in self.alpha_cuts.keys():
            l1, r1 = self.alpha_cuts[alpha]
            l2, r2 = other.alpha_cuts[alpha]
            new_cuts[alpha] = (np.amin([l1 + r2, l1 + l2, r1 + r2, r1 + l2]),
                               np.amax([l1 + r2, l1 + l2, r1 + r2, r1 + l2]))
                
        return FuzzyNumber(None,
                 self.membership_space,
                 None, new_cuts)
        
    def __radd__(self, other):
        """Handles right-side addition (e.g., 5 + FuzzyNumber)"""
        return self.__add__(other)
       
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            new_cuts = {}
            for alpha in self.alpha_cuts.keys():
                l, r = self.alpha_cuts[alpha]
                new_cuts[alpha] = np.amin([l - other, r - other]), np.amax([l - other, r - other])
            space_x_new = np.linspace(new_cuts[list(new_cuts.keys())[0]][0], new_cuts[list(new_cuts.keys())[0]][1], len(new_cuts))
            return FuzzyNumber(None,
                 self.membership_space,
                 space_x_new, new_cuts)
        elif not isinstance(other, FuzzyNumber):
            raise TypeError("Operations can only be performed between two FuzzyNumbers.")
        elif not np.array_equal(self.membership_space, other.membership_space):
            raise ValueError("FuzzyNumbers must have the same membership space for operations.")
        
        new_cuts = {}
        # Perform interval subtraction for each common alpha level
        for alpha in self.alpha_cuts.keys():
            l1, r1 = self.alpha_cuts[alpha]
            l2, r2 = other.alpha_cuts[alpha]
            new_cuts[alpha] = (np.amin([l1 - r2, l1 - l2, r1 - r2, r1 - l2]),
                               np.amax([l1 - r2, l1 - l2, r1 - r2, r1 - l2]))
                
        space_x_new = np.linspace(new_cuts[list(new_cuts.keys())[0]][0], new_cuts[list(new_cuts.keys())[0]][1], len(new_cuts))
        return FuzzyNumber(None,
                self.membership_space,
                space_x_new, new_cuts)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_cuts = {}
            for alpha in self.alpha_cuts.keys():
                l, r = self.alpha_cuts[alpha]
                new_cuts[alpha] = np.amin([l * other, r * other]), np.amax([l * other, r * other])
    
            space_x_new = np.linspace(new_cuts[list(new_cuts.keys())[0]][0], new_cuts[list(new_cuts.keys())[0]][1], len(new_cuts))
            return FuzzyNumber(None,
                 self.membership_space,
                 space_x_new, new_cuts)
        elif not isinstance(other, FuzzyNumber):
            raise TypeError("Operations can only be performed between two FuzzyNumbers.")
        elif not np.array_equal(self.membership_space, other.membership_space):
            raise ValueError("FuzzyNumbers must have the same membership space for operations.")
        
        new_cuts = {}
        # Perform interval multiplication for each common alpha level
        for alpha in self.alpha_cuts.keys():
            l1, r1 = self.alpha_cuts[alpha]
            l2, r2 = other.alpha_cuts[alpha]
            new_cuts[alpha] = (np.amin([l1 * l2, l1 * r2, r1 * l2, r1 * r2]),
                               np.amax([l1 * l2, l1 * r2, r1 * l2, r1 * r2]))
                
        return FuzzyNumber(None,
                 self.membership_space,
                 None, new_cuts)

    def __rmul__(self, other):
        """Handles right-side multiplication (e.g., 5 * FuzzyNumber)"""
        return self.__mul__(other)

    def plot(self, label=None):
        """Quick utility to plot the membership function"""
        plt.plot(self.space_x, self.membership_values, label="Membership function")
        plt.title(f"Membership function of fuzzy number {label}" if label else "Membership function of fuzzy number", fontsize=16)
        plt.axvline(self.center_of_gravity, color='blue', linestyle='--', label=f'CoG: {self.center_of_gravity:.3f}')
        plt.axvline(self.mean_of_maxima, color='green', linestyle='--', label=f'MoM {self.mean_of_maxima:.3f}')
        plt.legend(fontsize=14)
        plt.xlabel("Value", fontsize=16)
        plt.ylabel(r"$\alpha$ (Membership)", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        plt.show()
        
    # def __repr__(self):
    #     support = self.alpha_cuts.get(0.0, ("?", "?"))
    #     core = self.alpha_cuts.get(1.0, ("?", "?"))
    #     return f"FuzzyNumber(Support=[{support[0]:.2f}, {support[1]:.2f}], Core=[{core[0]:.2f}, {core[1]:.2f}])"