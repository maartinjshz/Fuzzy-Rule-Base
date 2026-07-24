import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import interp1d

def alpha_cut_all_intervals(xx, membership_vls, alpha, step_size):
    # Find where membership crosses alpha
    if alpha == 0:
        above = membership_vls > 0
    else:
        above = membership_vls >= alpha - 2 * step_size

    if not np.any(above):
        
        return alpha_cut_all_intervals(xx, membership_vls, alpha, 2*step_size)
    
    # this is required, as otherwise if alpha = 1 
    # Is just single point, it will be interpolated
    # to an interval, whichi will make problems later
    # calculating fuzzy functions. 
    if alpha == 1:
        exact_matches = np.where(membership_vls == 1.0)[0]
        if len(exact_matches) == 1:
            return [xx[exact_matches[0]], xx[exact_matches[0]]]
    
    indices = np.where(above)[0]
    left = indices[0]
    right = indices[-1]
    # Linear interpolation for more precise cut points
    def interp_edge(idx1, idx2):
        x1, x2 = xx[idx1], xx[idx2]
        y1, y2 = membership_vls[idx1], membership_vls[idx2]
        return x1 + (alpha - y1) * (x2 - x1) / (y2 - y1)
    # Left edge
    if left > 0:
        left_x = interp_edge(left-1, left)
    else:
        left_x = xx[left]
    # Right edge
    if right < len(xx)-1:
        right_x = interp_edge(right, right+1)
    else:
        right_x = xx[right]
    return [left_x, right_x]
 
def all_alpha_cuts(membership_vls, xx, alphas, step_size=1 / 500, n_jobs=1):
    """Compute alpha-cuts for all requested alpha levels.

    Parameters
    ----------
    membership_vls : np.ndarray
        Membership values of the fuzzy set.
    xx : np.ndarray
        X-axis values corresponding to ``membership_vls``.
    alphas : iterable
        Alpha levels for which to compute the cuts.
    step_size : float, optional
        Step used by ``alpha_cut_all_intervals`` when expanding the search.
    n_jobs : int, optional
        Number of worker threads to use. ``1`` keeps the original serial behavior.
    """
    alpha_list = list(alphas)
    if n_jobs is None or n_jobs <= 1:
        return {
            alpha: alpha_cut_all_intervals(xx, membership_vls, alpha, step_size)
            for alpha in alpha_list
        }

    def _compute_cut(alpha):
        return alpha, alpha_cut_all_intervals(xx, membership_vls, alpha, step_size)

    with ThreadPoolExecutor(max_workers=min(n_jobs, len(alpha_list))) as executor:
        results = executor.map(_compute_cut, alpha_list)

    return {alpha: cut for alpha, cut in results}


def create_1d_hat_function_scipy(grid_points, i, step_size = 1):
    """
    Creates a 1D triangle hat function phi_i(x) using scipy.interpolate.interp1d.

    Args:
        grid_points (np.array): A sorted 1D array of grid points.
        i (int): The index of the grid point for which to create the hat function.

    Returns:
        function: A function phi_i(x) that evaluates the hat function at x.
    """
    n = len(grid_points)
    if not (0 <= i < n):
        raise IndexError(f"Index i ({i}) is out of bounds for grid_points of length {n}")
    
    # Create y-values for the hat function: 1 at index i, 0 elsewhere
    y_values = np.zeros(n)
    y_values[i] = 1.0
    
    # Create the linear interpolation function
    # kind='linear' is crucial for triangle hat functions
    # fill_value=(0.0, 0.0) ensures it's 0 outside the defined range
    # bounds_error=False prevents error if x is outside range, instead uses fill_value
    hat_function = interp1d(grid_points, y_values, kind='linear',
                              fill_value=0.0, bounds_error=False)
    return hat_function

def create_1d_gaussian_function(grid_points, i, sigma_factor=0.5):
    """
    Creates a 1D Gaussian membership function mu_i(x).

    Args:
        grid_points (np.array): A sorted 1D array of grid points (the universe of discourse).
        i (int): The index of the grid point that serves as the center (mean) of the function.
        sigma_factor (float): Controls the spread/overlap. A larger factor means a narrower function.

    Returns:
        function: A function mu_i(x) that evaluates the Gaussian membership at x.
    """
    n = len(grid_points)
    if not (0 <= i < n):
        raise IndexError(f"Index i ({i}) is out of bounds for grid_points of length {n}")

    # 1. Determine the Mean (mu)
    # The center of the Gaussian is the grid point itself.
    mu = grid_points[i]

    # 2. Determine the Standard Deviation (sigma)
    # This dictates the spread and overlap with neighbors.
    if n > 1:
        # The distance between adjacent grid points (assuming uniform or taking the average/local step)
        # We use the average distance between points as a basis for spread.
        average_interval = np.mean(np.diff(grid_points))
        
        # Sigma is set relative to the interval to ensure good overlap.
        sigma = average_interval / sigma_factor
    else:
        # If only one point, define a reasonable sigma based on the scale of the domain
        sigma = (grid_points[-1] - grid_points[0]) / 10 if grid_points.size > 0 else 1.0


    # 3. Define and return the Gaussian function (closure)
    def gaussian_mf(x):
        # Calculate the Gaussian membership degree using the formula:
        # mu(x) = exp(-1/2 * ((x - mu) / sigma)^2)
        
        # Ensure x is a numpy array for vectorized calculation
        x_array = np.asarray(x)
        
        # Handle the case where sigma might be zero (though unlikely with the above calculation)
        if sigma == 0:
            return np.where(x_array == mu, 1.0, 0.0)

        # Calculate the exponent
        exponent = -0.5 * ((x_array - mu) / sigma)**2
        
        return np.exp(exponent)

    return gaussian_mf

def reconstruct_curve_from_given_alpha_cut_intervals(alpha_cuts, x_grid):
    pass

def reconstruct_curve_from_alpha_cuts(alpha_cuts, x_grid):
    """Reconstruct a fuzzy membership curve from alpha-cut intervals.

    This version avoids repeated per-alpha boolean-mask scans by computing
    the relevant index ranges directly from the interval endpoints and then
    filling the curve in a single pass.
    """
    if not alpha_cuts:
        return np.zeros_like(x_grid, dtype=float)

    sorted_alphas = sorted(alpha_cuts.keys(), reverse=True)
    values = np.array([list(v) for v in alpha_cuts.values()], dtype=float)
    mu_reconstructed = np.zeros_like(x_grid, dtype=float)
    
    current_x_grid = np.array([*values[:,1], *values[:,0]])
    mu_reconstructed = np.array(
    [*sorted(alpha_cuts.keys()),
     *sorted_alphas]
)
     
    # for alpha in sorted_alphas:
    #     interval = alpha_cuts[alpha]
    #     if not isinstance(interval, (list, tuple)) or len(interval) != 2:
    #         continue

    #     left, right = float(np.min(interval)), float(np.max(interval))
    #     if not np.isfinite(left) or not np.isfinite(right):
    #         continue

    #     mask = (x_grid >= left) & (x_grid <= right)
    #     if not np.any(mask):
    #         continue

    #     indices = np.flatnonzero(mask)
    #     leftmost = indices[0]
    #     rightmost = indices[-1]

    #     mu_reconstructed[leftmost] = max(mu_reconstructed[leftmost], alpha)
    #     mu_reconstructed[rightmost] = max(mu_reconstructed[rightmost], alpha)

    nonzero_idx = np.flatnonzero(mu_reconstructed != 0)
    if nonzero_idx.size == 0:
        return np.zeros_like(x_grid, dtype=float)

    return np.interp(x_grid, current_x_grid[nonzero_idx], mu_reconstructed[nonzero_idx], left=0, right=0)

def centroid_from_curve(x_grid, mu):
    numerator = np.trapezoid(x_grid * mu, x_grid)
    denominator = np.trapezoid(mu, x_grid)
    return numerator / denominator if denominator != 0 else 0


def reconstruct_curve(payoff_fun_value, number_of_grid_points = 500):
    
    payoff_fun_value_intervals= {alpha: [(vals[0], vals[1])] for alpha, vals in payoff_fun_value.items()}
    x_grid = np.linspace(
        min([interval[0] for intervals in payoff_fun_value_intervals.values() for interval in intervals]),
        max([interval[1] for intervals in payoff_fun_value_intervals.values() for interval in intervals]),
        number_of_grid_points
    )
    # print(payoff_fun_value_intervals)
    mu_curve = reconstruct_curve_from_alpha_cuts(payoff_fun_value_intervals, x_grid)
    if np.amax(mu_curve) == 0:
        return 0, mu_curve, x_grid
    cog = centroid_from_curve(x_grid, mu_curve)
 
    return cog, mu_curve, x_grid
