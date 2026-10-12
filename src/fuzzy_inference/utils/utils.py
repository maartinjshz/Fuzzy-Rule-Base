import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter

def update_permutation(permutations, i, max_value=5):
    """Function to move to the next permutation of rules. 

    Args:
        permutations (List): what permutation of rules is to active
        i (id): to which element +1 should be added
        max_value (int): the maximum value for each element in the permutation

    Returns:
        permutations (List):  returns list of active rules for next  iteration
        flag (Bool): if True, all combinations have been ran. 
    """    
    flag = False
    if i == len(permutations):
        return permutations, True
    permutations[i] += 1
    
    if permutations[i] == max_value:
        permutations[i] = 0
        permutations, flag = update_permutation(permutations, i + 1)
        
    return permutations, flag


def make_price_analysis_as_gif(
    results_matrix,
    x_vals,
    y_vals,
    middle_vals=None,
    out_path="payoff_middle_dim.gif",
    fps=2,
    save_indices=None,
    frame_dir="saved_frames"
):
    if save_indices is None:
        save_indices = [0, 4, 8, 10]

    os.makedirs(frame_dir, exist_ok=True)

    n_frames = results_matrix.shape[1]
    if middle_vals is None:
        middle_vals = np.arange(1, n_frames + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    data0 = results_matrix[:, 0, :].T
    im = ax.imshow(
        data0,
        origin="lower",
        aspect="auto",
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
        cmap="plasma"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Payoff value")

    ax.set_xlabel("Payoff value")
    ax.set_ylabel("Cleaning C_2 costs")
    title = ax.set_title(f"Frame 1/{n_frames} (middle={middle_vals[0]})")

    def update(frame_idx):
        frame_data = results_matrix[:, frame_idx, :].T
        im.set_data(frame_data)
        im.set_clim(vmin=np.min(results_matrix), vmax=np.max(results_matrix))
        title.set_text(f"Frame {frame_idx+1}/{n_frames} (middle={middle_vals[frame_idx]})")

        # Save selected frames as standalone PNGs
        if frame_idx in save_indices:
            fig.savefig(
                os.path.join(frame_dir, f"frame_{frame_idx}.png"),
                dpi=200,
                bbox_inches="tight"
            )

        return im, title

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 // max(fps, 1),
        blit=False
    )
    ani.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved GIF: {out_path}")
    print(f"Saved selected frames: {save_indices}")