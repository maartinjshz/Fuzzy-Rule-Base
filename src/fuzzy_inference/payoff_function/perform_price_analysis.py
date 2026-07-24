import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

from fuzzy_inference.fuzzy_utils.fuzzy_sets   import FuzzyNumber


def call_payoff(payoff_formula, u, mu, **kwargs):
    print(u)
    if np.isscalar(u):
        return payoff_formula(u, mu, **kwargs)
    return payoff_formula(*u, mu, **kwargs)

def full_voi_analysis(
    membership_value, 
    step_size, 
    param_config,         
    payoff_formula,       
    grid_points=100, 
    u_space=np.linspace(0, 1, 100)
):
    """Function for performing full Voi analysis.
    
    arguments:
        membership_value (FuzzyNumber): Fuzzy membership value from inference.
        step_size (int): Step size for parameter ranges.
        param_config (dict): Configuration for parameters, including bounds and width of support.
        payoff_formula (callable): Payoff function to use.
        grid_points (int): Number of grid points for fuzzy sets (on Y axis). 
    Returns:
        tuple: A tuple containing the results matrix and parameter ranges.
    
    """
    param_names = list(param_config.keys())
    param_ranges = {}
    
    # 1. Parse bounds and generate 1D ranges for each parameter
    for name, config in param_config.items():
        p_min, p_max = config['bounds']
        steps = int((p_max - p_min) / step_size)
        steps = max(steps, 1) 
        param_ranges[name] = np.linspace(p_min, p_max, steps)
    
    # 2. Dynamically create the N-dimensional results array
    matrix_shape = tuple(len(param_ranges[name]) for name in param_names)
    results_MOM = np.zeros(matrix_shape)
    
    enumerated_ranges = [list(enumerate(param_ranges[name])) for name in param_names]
    # u_space_arr = np.asarray(u_space, dtype=float)
    membership_space_x = np.asarray(membership_value.space_x, dtype=float)

    # u_space = np.asarray(u_space)

        # 1. Standardize into a sequence (1D becomes a 1-element tuple, N-D stays a tuple/list)
    if isinstance(u_space, np.ndarray) and u_space.dtype == object:
        # Fixes the case where ogrid got wrapped in np.array()
        u_seq = tuple(u_space)
    elif isinstance(u_space, np.ndarray) and u_space.ndim > 1:
        # Fixes np.mgrid dense arrays
        u_seq = tuple(u_space)
    elif isinstance(u_space, (list, tuple)):
        # Fixes pure np.ogrid lists
        u_seq = tuple(u_space)
    else:
        # Standard 1D np.linspace
        u_seq = (u_space,)

    total_count = np.broadcast(*u_seq).size

    # 3. Iterate through all parameter combinations
    total_combinations = np.prod([len(r) for r in enumerated_ranges])

    for combo in tqdm(itertools.product(*enumerated_ranges),
                    total=total_combinations,
                    desc="Evaluating parameter grid"):
        indices = tuple(c[0] for c in combo)
        values = [c[1] for c in combo]
        
        # Current crisp scalar values
        crisp_params = dict(zip(param_names, values))
        
        # 5. Perfect VoI (Crisp calculation)
        perfect_voi_val = np.empty(len(membership_value.membership_values), dtype=float)
        for id_a, a in enumerate(membership_value.membership_values):
 
            payoff_values = np.fromiter(
                                (
                                    payoff_formula(*u_i, membership_space_x[id_a], **crisp_params) 
                                    for u_i in np.broadcast(*u_seq)  
                                ),
                                dtype=float,
                                count=total_count, # Safer than len(u_samples) for N-D grids
                            )
            perfect_voi_val[id_a] = a * np.amax(payoff_values)
        
        # 6. Prior Knowledge Payoff (Fuzzy calculation)
        prior_knowledge_payoff = np.fromiter(
            (payoff_formula(*u_i, membership_value, **crisp_params).mean_of_maxima for u_i in np.broadcast(*u_seq)),
            dtype=float,
            count=total_count,
        )
        
        # Save output to its corresponding coordinate block
        V_i = np.amax(perfect_voi_val) - np.amax(prior_knowledge_payoff)
        results_MOM[indices] = V_i 
        
    return results_MOM, param_ranges

# def check_u_space_size(u):
#     if np.isscalar(u):
#         return u
#     else:
#         return *u

def perfect_additional_VoI_analysis(
    membership_value, 
    additional_info,
    step_size, 
    param_config,         
    payoff_formula,       
    grid_points=100, 
    u_space=np.linspace(0, 1, 100)
):
    """Function for performing full Voi analysis.
    
    arguments:
        membership_value (FuzzyNumber): Fuzzy membership value from inference.
        step_size (int): Step size for parameter ranges.
        param_config (dict): Configuration for parameters, including bounds and width of support.
        payoff_formula (callable): Payoff function to use.
        grid_points (int): Number of grid points for fuzzy sets (on Y axis). 
    Returns:
        tuple: A tuple containing the results matrix and parameter ranges.
    
    """
    param_names = list(param_config.keys())
    param_ranges = {}
    
    # 1. Parse bounds and generate 1D ranges for each parameter
    for name, config in param_config.items():
        p_min, p_max = config['bounds']
        steps = int((p_max - p_min) / step_size)
        steps = max(steps, 1) 
        param_ranges[name] = np.linspace(p_min, p_max, steps)
    
    # 2. Dynamically create the N-dimensional results array
    matrix_shape = tuple(len(param_ranges[name]) for name in param_names) 
    results_MOM = np.zeros(matrix_shape)
    result_perfect_info = np.zeros(matrix_shape)
    results_additional_info = np.zeros(matrix_shape)
    
    enumerated_ranges = [list(enumerate(param_ranges[name])) for name in param_names]

    
    print(f"Running multidimensional analysis for parameters: {param_names}")
    print(f"Grid shape: {matrix_shape}")

    # 3. Iterate through all parameter combinations
    for combo in itertools.product(*enumerated_ranges):
        indices = tuple(c[0] for c in combo)
        values = [c[1] for c in combo]
        
        # Current crisp scalar values
        crisp_params = dict(zip(param_names, values))
        
        # 4. Generate custom, asymmetrical fuzzy objects dynamically
        fuzzy_params = {}
        for name, val in crisp_params.items():
            # Extract custom left/right spreads (default to 0 if not provided)
            left_sp = param_config[name].get('left_spread', 0)
            right_sp = param_config[name].get('right_spread', 0)
            
            # Asymmetrical bounds: [Peak - Left, Peak, Peak + Right]
            fuzzy_params[name] = FuzzyNumber.triangular(
                val - left_sp, 
                val, 
                val + right_sp, 
                np.linspace(0, 1, grid_points)
            )
        
        # 5. Perfect VoI (Crisp calculation)
        perfect_voi_max = 0.0
        for id_a, a in enumerate(membership_value.membership_values):
            payoff_values = np.fromiter(
                (payoff_formula(float(u_i), membership_value.space_x[id_a], **crisp_params) for u_i in np.nditer(u_space)),
                dtype=float,
                count=len(u_space),
            )
            perfect_voi_max = max(perfect_voi_max, a * np.amax(payoff_values))
        
        # 6. Additional information Payoff (Fuzzy calculation)
        additional_knowledge_max = 0.0
        for id_b, b in enumerate(additional_info.membership_values):
            new_info_membership_fun = FuzzyNumber(
                membership_values=np.minimum(membership_value.membership_values, b) / b,
                membership_space=membership_value.membership_space,
                space_x=membership_value.space_x,
            )
            payoff_values = np.fromiter(
                (call_payoff(payoff_formula, u_i, new_info_membership_fun, **crisp_params).mean_of_maxima for u_i in np.nditer(u_space)),
                dtype=float,
                count=len(u_space),
            )
            additional_knowledge_max = max(additional_knowledge_max, b * np.amax(payoff_values))

        result_perfect_info[indices] = perfect_voi_max
        results_additional_info[indices] = additional_knowledge_max

    return result_perfect_info, results_additional_info, param_ranges



def plot_payoff_plots(data, plot_title, x_i_vals, y_i_vals, axis_titles = ["xi", "yi"]):
    """Create heatmap visualization of payoff values.

    Args:
        data (np.ndarray): NxN array of payoff values.
        plot_title (str): Title for the plot.
        x_i_vals (np.ndarray): Grid values for x_i parameter (X axis).
        y_i_vals (np.ndarray): Grid values for y_i parameter (Y axis).
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Calculate extent for correct axis scaling and labeling
    x_i_step = x_i_vals[1] - x_i_vals[0]
    y_i_step = y_i_vals[1] - y_i_vals[0]

    extent = [
        x_i_vals[0] - x_i_step / 2,     # xmin
        x_i_vals[-1] + x_i_step / 2,    # xmax
        y_i_vals[0] - y_i_step / 2,     # ymin
        y_i_vals[-1] + y_i_step / 2     # ymax
    ]

    im = ax.imshow(
        data,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='plasma'
    )
    fig.colorbar(im, ax=ax, label='Calculated MoM Value')
    ax.set_title(plot_title, fontsize=21)
    ax.set_xlabel(f'{axis_titles[0]}', fontsize=16)
    ax.set_ylabel(f'{axis_titles[1]}', fontsize=16)
    ax.set_xticks(x_i_vals[::20])
    ax.set_yticks(y_i_vals[::20])
    ax.grid(which='major', color='white', linestyle=':', linewidth=0.5, alpha=0.5)

    fig.suptitle('Price analysis for different Costs', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
 