import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from matplotlib.ticker import FuncFormatter

from config import RANK_TOL 
from geometry import fisher


def plot_comprehensive(params, X, y, loss_history, acc_history, ricci_history, 
                      rank_history, epochs_list, eigenvalues, output_dir, kretschmann_history, weyl_history, size=7, **kwargs):
    """Generate and save individual square plots at 500 dpi in PDF format."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def save_square_plot(x, y, title, xlabel, ylabel, filename, logscale=False, hline=None, symlog=False, 
                         dots=None, linthresh=1, custom_ticks=False):
        fig, ax = plt.subplots(figsize=(size, 5))
        
        if symlog:  # Symmetric log scale for data with positive and negative values
            # Plot with regular line
            ax.plot(x, y, '-', linewidth=1.5, color='darkblue')
            ax.set_yscale('symlog', linthresh=linthresh)
            
            # Add custom ticks for symlog if requested
            if custom_ticks:
                # Find the maximum absolute value in the data
                abs_max = max(abs(np.max(y)), abs(np.min(y)))
                
                if abs_max > 0:
                    # Determine the appropriate range based on data magnitude
                    log_range = int(np.ceil(np.log10(abs_max)))
                    
                    # Generate custom tick positions with wider spacing
                    pos_ticks = [10**i for i in range(-log_range, log_range+1, 2) if 10**i <= abs_max]
                    
                    # Filter out small values too close to zero to avoid overlaps
                    min_tick = 1e-1  # Minimum threshold to avoid ticks too close to zero
                    pos_ticks = [x for x in pos_ticks if x >= min_tick]
                    neg_ticks = [-x for x in pos_ticks]  # Mirror for negative values
                    
                    # Combine ticks with zero, ensuring sorted order
                    ticks = sorted(list(set([0] + neg_ticks + pos_ticks)))
                    
                    # Set custom tick positions and format them
                    ax.set_yticks(ticks)
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: 
                        "0" if abs(x) < 1e-10 else  # Handle exact zero and near-zero
                        (f"{x:.1f}" if 0 < abs(x) < 1 else f"{x:.0e}")))
            
        elif logscale:
            ax.semilogy(x, y, '-', linewidth=1.5, alpha=0.7, color='darkblue')  # Line plot for logscale
        else:
            ax.plot(x, y, '-', linewidth=1.5, color='darkblue')  # Line plot
            
        if hline is not None:
            ax.axhline(y=hline[0], color=hline[1], linestyle='--', label=hline[2])
            ax.legend(fontsize=14)  # Set legend font size
            
        if dots is not None:
            if logscale:
                ax.semilogy(np.arange(len(x)), y, 'o-', color='darkblue')
            else:
                ax.plot(np.arange(len(x)), y, 'o-', color='darkblue')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)  # Set x-axis font size
        ax.set_ylabel(ylabel, fontsize=12)  # Set y-axis font size
        ax.tick_params(axis='both', labelsize=14)  # Set tick label font size
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{filename}_{timestamp}.pdf"), dpi=500, bbox_inches='tight')
        plt.close(fig)

    # 1. Loss history
    save_square_plot(epochs_list, loss_history,
                     title="Training Loss",
                     xlabel="Epochs", ylabel="Loss",
                     filename="plot_loss", logscale=True)

    # 2. Accuracy
    save_square_plot(epochs_list, acc_history,
                     title="Accuracy",
                     xlabel="Epochs", ylabel="Accuracy",
                     filename="plot_accuracy")

    # 3. Rank
    save_square_plot(epochs_list, rank_history,
                     title="Fisher Matrix Rank",
                     xlabel="Epochs", ylabel="Rank",
                     filename="plot_rank")

    # 4. Eigenvalues
    sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    save_square_plot(np.arange(len(sorted_eigenvalues)), sorted_eigenvalues,
                     title="Eigenvalue Spectrum",
                     xlabel="Eigenvalue Index", ylabel="Magnitude (Log Scale)",
                     filename="plot_eigenvalues",
                     logscale=True,
                     hline=(RANK_TOL, 'r', f'Tol = {RANK_TOL:.0e}'), dots=True)

    # 5. Heatmap (special case)
    fisher_matrix = fisher(params, X)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(np.log10(np.abs(fisher_matrix) + 1e-16),
                   cmap='viridis', interpolation='nearest')
    ax.set_title("Fisher Information Matrix (Log10 Scale)", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)  # Set tick label font size
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"plot_fisher_heatmap_{timestamp}.pdf"), dpi=500, bbox_inches='tight')
    plt.close(fig)

    # 6. Ricci Scalar with symmetric log scale and improved ticks
    save_square_plot(epochs_list, np.abs(ricci_history),
                     title="Ricci Scalar",
                     xlabel="Epochs", ylabel="Ricci Scalar (symlog scale)",
                     filename="plot_ricci",
                     logscale=True,
                     )
    
    # 7. Kretschmann Scalar with symmetric log scale
    save_square_plot(epochs_list, kretschmann_history,
                     title="Kretschmann Scalar",
                     xlabel="Epochs", ylabel="Kretschmann Scalar (symlog scale)",
                     filename="plot_kretschmann",
                     logscale=True
                    )

    # 8. Weyl Scalar with symmetric log scale
    save_square_plot(epochs_list, weyl_history,
                     title="Weyl Scalar",
                     xlabel="Epochs", ylabel="Weyl Scalar (symlog scale)",
                     filename="plot_weyl",
                     logscale=True)


    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"plot_decision_and_fisher_{timestamp}.pdf"),
                dpi=500, bbox_inches='tight')
    plt.close(fig)
    
def show_all_plots_together(loss_history, acc_history, rank_history, epochs_list,
                        eigenvalues, ricci_history, fisher_matrix, output_dir, kretschmann_history, weyl_history, **kwargs):
    """Show and save all plots in a single combined figure."""
    from datetime import datetime
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.ticker import FuncFormatter, LogLocator, SymmetricalLogLocator, MaxNLocator
    
    # Get rank_tol from kwargs or use default    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a figure with GridSpec for better layout control
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3)
    
    # Improve y-axis ticks to show signed magnitude
    def signed_log_format(y, _):
        if y == 0:
            return "0"
        sign = "-" if y < 0 else ""
        return f"${sign}10^{{{int(np.log10(abs(y)))}}}$"

    # 1. Loss (Top row, first column)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(epochs_list, loss_history, '-', color='darkblue')
    ax1.set_title("Loss", fontsize=14)
    ax1.set_xlabel("Epochs", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_ylim(0,3)
    ax1.grid(True)

    # 2. Accuracy (Top row, middle column)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_list, acc_history, '-', color='darkblue')
    ax2.set_title("Accuracy", fontsize=14)
    ax2.set_xlabel("Epochs", fontsize=14)
    ax2.set_ylabel("Accuracy", fontsize=14)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.grid(True)

    # 3. Rank (Top row, last column)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs_list, rank_history, '-', color='darkblue')
    ax3.set_title("Reduced Fisher Metric rank", fontsize=14)
    ax3.set_xlabel("Epochs", fontsize=14)
    ax3.set_ylabel("Rank", fontsize=14)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.grid(True)

    # 4. Eigenvalues (Middle row, first column)
    ax4 = fig.add_subplot(gs[1, 0])
    sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
    ax4.semilogy(np.arange(len(sorted_eigs)), sorted_eigs, 'o-', color='darkblue')
    ax4.axhline(RANK_TOL, color='r', linestyle='--', label=f'Tol={RANK_TOL:.0e}')
    ax4.set_title("Eigenvalue Spectrum", fontsize=14)
    ax4.set_xlabel("Index", fontsize=14)
    ax4.set_ylabel("Magnitude", fontsize=14)
    ax4.tick_params(axis='both', labelsize=14)
    ax4.legend(fontsize=14)
    ax4.grid(True)

    # 5. Ricci Scalar (Bottom row, spans all columns)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.semilogy(epochs_list, np.abs(ricci_history), '-', color='darkblue')
    ax5.set_title("Ricci Scalar", fontsize=14)
    ax5.set_xlabel("Epochs", fontsize=14)
    ax5.set_ylabel("Ricci Scalar (symlog)", fontsize=14)
    ax5.tick_params(axis='both', labelsize=14)
    ax5.grid(True)

    # 6. Fisher heatmap (Middle row, last column)
    ax6 = fig.add_subplot(gs[1, 2])
    im = ax6.imshow(np.log10(np.abs(fisher_matrix) + 1e-16),
                     cmap='viridis', interpolation='nearest')
    ax6.set_title("Fisher Matrix magnitudes (log10)", fontsize=14)
    ax6.tick_params(axis='both', labelsize=14)
    fig.colorbar(im, ax=ax6)

    # Combined Kretschmann and Weyl Scalars (Middle row, middle column)
    ax8 = fig.add_subplot(gs[1, 1])
    ax8.semilogy(epochs_list, kretschmann_history, '-', label='Kretschmann', color='navy')
    ax8.semilogy(epochs_list, weyl_history, '--', label='Weyl', color='magenta')
    ax8.set_title("Curvature Scalars", fontsize=14)
    ax8.set_xlabel("Epochs", fontsize=14)
    ax8.set_ylabel("Value (log)", fontsize=14)
    ax8.tick_params(axis='both', labelsize=14)
    ax8.legend(fontsize=12)
    ax8.grid(True)

    # Save the combined figure
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"all_plots_{timestamp}.pdf")
    plt.savefig(fig_path, dpi=500, bbox_inches='tight')
    
    # Show the figure
    plt.show()

def plot_eigenvalues_over_time(epochs_list, eigenvalues_history, output_dir):

    # Convert eigenvalues_history to a NumPy array for easier slicing
    eigenvalues_history = np.array(eigenvalues_history)

    # Plot the largest eigenvalue over time
    largest_eigenvalues = eigenvalues_history[:, -1]  # Last column (largest eigenvalue)
    plt.figure(figsize=(8, 6))
    plt.semilogy(epochs_list, largest_eigenvalues, label="Largest Eigenvalue")
    plt.title("Largest Eigenvalue of Fisher Matrix Over Time", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Eigenvalue", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "largest_eigenvalue_over_time.pdf"), dpi=300)
    plt.close()

    # Plot the full spectrum at selected epochs
    for i, epoch in enumerate(epochs_list):
        plt.figure(figsize=(8, 6))
        plt.semilogy(np.arange(len(eigenvalues_history[i])), eigenvalues_history[i], 'o-', label=f"Epoch {epoch}")
        plt.title(f"Fisher Eigenvalue Spectrum at Epoch {epoch}", fontsize=14)
        plt.xlabel("Eigenvalue Index", fontsize=12)
        plt.ylabel("Eigenvalue", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"eigenvalue_spectrum_epoch_{epoch}.pdf"), dpi=300)
        plt.close()

import matplotlib.animation as animation

def create_eigenvalue_slideshow(epochs_list, eigenvalues_history, output_dir):

    # Convert eigenvalues_history to a NumPy array for easier slicing
    eigenvalues_history = np.array(eigenvalues_history)

    # Create a figure for the animation
    fig, ax = plt.subplots(figsize=(8, 6))

    # Initialize the plot
    def init():
        ax.clear()
        ax.set_title("Eigenvalue Spectrum Over Epochs", fontsize=14)
        ax.set_xlabel("Eigenvalue Index", fontsize=12)
        ax.set_ylabel("Eigenvalue (Log Scale)", fontsize=12)
        ax.set_ylim(1e-20, 1e2)  # Fix y-axis range between 10^0 and 10^-18        
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        return ax,

    # Update the plot for each frame
    def update(frame):
        ax.clear()
        ax.set_title(f"Eigenvalue Spectrum at Epoch {epochs_list[frame]}", fontsize=14)
        ax.set_xlabel("Eigenvalue Index", fontsize=12)
        ax.set_ylabel("Eigenvalue (Log Scale)", fontsize=12)
        ax.set_yscale("log")
        ax.set_ylim(1e-20, 1e2)  # Fix y-axis range between 10^0 and 10^-18        
        ax.grid(True, alpha=0.3)
        eigenvalues = eigenvalues_history[frame]
        ax.semilogy(np.arange(len(eigenvalues)), eigenvalues, 'o-', label=f"Epoch {epochs_list[frame]}")
        ax.legend(fontsize=12)
        return ax,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(epochs_list), init_func=init, blit=False)

    # Save the animation as an MP4 file
    animation_path = os.path.join(output_dir, "eigenvalue_spectrum_animation.gif")
    ani.save(animation_path, writer="pillow", fps=2)
    print(f"Animation saved to {animation_path}")