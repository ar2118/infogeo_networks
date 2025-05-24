"""
Visualization module for XOR spiral task.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime
from config import RANK_TOL as rank_tol
from geometry import fisher
from model import forward


def plot_comprehensive(params, X, y, centers, colors, loss_history, acc_history, ricci_history, 
                      rank_history, epochs_list, eigenvalues, output_dir, kretschmann_history, 
                      weyl_history, size=7, **kwargs):
    """Generate and save individual square plots at 500 dpi in PDF format."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def save_square_plot(x, y, title, xlabel, ylabel, filename, logscale=False, hline=None, 
                         dots=None):
        fig, ax = plt.subplots(figsize=(size, 5))
            
        if logscale:
            ax.semilogy(x, y, '-', linewidth=1.5, alpha=0.7, color='darkblue')
        else:
            ax.plot(x, y, '-', linewidth=1.5, color='darkblue')
            
        if hline is not None:
            ax.axhline(y=hline[0], color=hline[1], linestyle='--', label=hline[2])
            ax.legend(fontsize=14)
            
        if dots is not None:
            if logscale:
                ax.semilogy(np.arange(len(x)), y, 'o-', color='darkblue')
            else:
                ax.plot(np.arange(len(x)), y, 'o-', color='darkblue')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='both', labelsize=14)
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
                     hline=(rank_tol, 'r', f'Tol = {rank_tol:.0e}'), dots=True)

    # 5. Fisher Matrix Heatmap
    fisher_matrix = fisher(params, X)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(np.log10(np.abs(fisher_matrix) + 1e-16),
                   cmap='viridis', interpolation='nearest')
    ax.set_title("Fisher Information Matrix (Log10 Scale)", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"plot_fisher_heatmap_{timestamp}.pdf"), dpi=500, bbox_inches='tight')
    plt.close(fig)

    # 6. Ricci Scalar
    save_square_plot(epochs_list, np.abs(ricci_history),
                     title="Ricci Scalar (Log Scale, Negative Values)",
                     xlabel="Epochs", ylabel="Ricci Scalar (Log Scale)",
                     filename="plot_ricci",
                     logscale=True)
    
    # 7. Kretschmann Scalar
    save_square_plot(epochs_list, kretschmann_history,
                     title="Kretschmann Scalar",
                     xlabel="Epochs", ylabel="Kretschmann Scalar (Log Scale)",
                     filename="plot_kretschmann",
                     logscale=True)

    # 8. Weyl Scalar
    save_square_plot(epochs_list, weyl_history,
                     title="Weyl Scalar",
                     xlabel="Epochs", ylabel="Weyl Scalar (Log Scale)",
                     filename="plot_weyl",
                     logscale=True)

    # 9. Decision boundary plot
    fig, ax_decision = plt.subplots(figsize=(7, 5))

    # Create a grid for the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict class for each point in the grid
    Z = forward(params, grid_points)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax_decision.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    ax_decision.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k', s=50)

    # Draw quadrant lines
    ax_decision.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax_decision.axvline(x=0, color='black', linestyle='--', alpha=0.3)

    # Plot cluster centers if any
    for i, center in enumerate(centers):
        ax_decision.plot(center[0], center[1], 'k*', markersize=15)
        ax_decision.annotate(f"Cluster {i}", (center[0], center[1]),
                             xytext=(10, 10), textcoords='offset points')

    ax_decision.set_xlim(xx.min(), xx.max())
    ax_decision.set_ylim(yy.min(), yy.max())
    ax_decision.set_title("Decision Boundary", fontsize=14)
    ax_decision.set_xlabel("Feature 1", fontsize=12)
    ax_decision.set_ylabel("Feature 2", fontsize=12)
    ax_decision.tick_params(axis='both', labelsize=12)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=10, label='Class 0 [1,0]'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=10, label='Class 1 [0,1]')
    ]
    ax_decision.legend(handles=legend_elements, fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"plot_decision_and_fisher_{timestamp}.pdf"),
                dpi=500, bbox_inches='tight')
    plt.close(fig)


def show_all_plots_together(loss_history, acc_history, rank_history, epochs_list,
                           eigenvalues, ricci_history, fisher_matrix, output_dir, 
                           kretschmann_history, weyl_history, **kwargs):
    """Show and save all plots in a single combined figure."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a figure with GridSpec for better layout control
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3)

    # 1. Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(epochs_list, loss_history, '-', color='darkblue')
    ax1.set_title("Loss", fontsize=14)
    ax1.set_xlabel("Epochs", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_ylim(0, 3)
    ax1.grid(True)

    # 2. Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_list, acc_history, '-', color='darkblue')
    ax2.set_title("Accuracy", fontsize=14)
    ax2.set_xlabel("Epochs", fontsize=14)
    ax2.set_ylabel("Accuracy", fontsize=14)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.grid(True)

    # 3. Rank
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs_list, rank_history, '-', color='darkblue')
    ax3.set_title("Reduced Fisher Metric rank", fontsize=14)
    ax3.set_xlabel("Epochs", fontsize=14)
    ax3.set_ylabel("Rank", fontsize=14)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.grid(True)

    # 4. Eigenvalues
    ax4 = fig.add_subplot(gs[1, 0])
    sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
    ax4.semilogy(np.arange(len(sorted_eigs)), sorted_eigs, 'o-', color='darkblue')
    ax4.axhline(rank_tol, color='r', linestyle='--', label=f'Tol={rank_tol:.0e}')
    ax4.set_title("Eigenvalue Spectrum", fontsize=14)
    ax4.set_xlabel("Index", fontsize=14)
    ax4.set_ylabel("Magnitude", fontsize=14)
    ax4.tick_params(axis='both', labelsize=14)
    ax4.legend(fontsize=14)
    ax4.grid(True)

    # 5. Ricci Scalar
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.semilogy(epochs_list, np.abs(ricci_history), '-', color='darkblue')
    ax5.set_title("Ricci Scalar", fontsize=14)
    ax5.set_xlabel("Epochs", fontsize=14)
    ax5.set_ylabel("Ricci Scalar (Abs)", fontsize=14)
    ax5.tick_params(axis='both', labelsize=14)
    ax5.grid(True)

    # 6. Fisher heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    im = ax6.imshow(np.log10(np.abs(fisher_matrix) + 1e-16),
                    cmap='viridis', interpolation='nearest')
    ax6.set_title("Fisher Matrix magnitudes (log10)", fontsize=14)
    ax6.tick_params(axis='both', labelsize=14)
    fig.colorbar(im, ax=ax6)

    # 7. Combined Kretschmann and Weyl Scalars
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
    """Plot eigenvalues evolution over training epochs."""
    
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


def create_eigenvalue_slideshow(epochs_list, eigenvalues_history, output_dir):
    """Create an animated slideshow of eigenvalue evolution."""
    
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
        ax.set_ylim(1e-20, 1e2)  # Fix y-axis range
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
        ax.set_ylim(1e-20, 1e2)  # Fix y-axis range
        ax.grid(True, alpha=0.3)
        eigenvalues = eigenvalues_history[frame]
        ax.semilogy(np.arange(len(eigenvalues)), eigenvalues, 'o-', label=f"Epoch {epochs_list[frame]}")
        ax.legend(fontsize=12)
        return ax,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(epochs_list), init_func=init, blit=False)

    # Save the animation as a GIF file
    animation_path = os.path.join(output_dir, "eigenvalue_spectrum_animation.gif")
    ani.save(animation_path, writer="pillow", fps=2)
    print(f"Animation saved to {animation_path}")