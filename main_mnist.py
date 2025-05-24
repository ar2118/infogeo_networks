import sys
if sys.platform == "win32":
    import types
    sys.modules['resource'] = types.ModuleType('resource')  # Fake empty module

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time
from datetime import datetime
import os
from tqdm import trange

from config import *
from data_generation_mnist import load_mnist_8_9, subsample_x
from training_mnist import init_params_10_hidden, train_epoch, loss_fn, accuracy, init_params_10_hidden, count_parameters
from geometry import compute_ricci_tensor_from_fisher, fisher, calculate_rank
from visuals_mnist import plot_comprehensive, show_all_plots_together, create_eigenvalue_slideshow

jax.config.update("jax_enable_x64", True)

def main(number_epoch=NUMBER_EPOCH, learning_rate=LEARNING_RATE, key_manual_check=False, no_ricci = NO_RICCI):
    """Main training loop for MNIST 8/9 classification task."""
    
    # Generate data
    X, y = load_mnist_8_9()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"MNIST4_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize parameters
    if key_manual_check:
        key = random.PRNGKey(KEY_MANUAL)
    else:
        key = random.PRNGKey(int(time.time()))  # Use the current time as the seed
    
    params = init_params_10_hidden(key, hidden_sizes=HIDDEN_SIZES)

    total_params = count_parameters(params)
    print(f"Total parameters: {total_params}")    
    
    # Training hyperparameters
    epochs = number_epoch
    lr = learning_rate
    
    # History trackers
    loss_history = []
    acc_history = []
    rank_history = []
    epochs_list = []
    ricci_history = []
    ricci_epochs = []
    eigenvalues_history = []
    kretschmann_history = []
    weyl_history = []
    
    # Initialize constant projection matrices
    constant_projection_matrix = None
    constant_projection_matrix_T = None
    
    # Training loop
    progress = trange(epochs, desc="Training", leave=True)
    for epoch in progress:
        params = train_epoch(params, X, y, lr, batch_size=32)
        
        # Record metrics every 100 epochs
        if epoch > 50 and epoch % 100 == 0 or epoch == epochs - 1:
            loss = loss_fn(params, X, y)
            acc = accuracy(params, X, y)
            
            # Convert JAX arrays to Python floats
            loss_value = float(loss)
            acc_value = float(acc)
            
            loss_history.append(loss_value)
            acc_history.append(acc_value)
            
            fisher_matrix = fisher(params, X)

            # Compute projection matrices dynamically before the 50000th epoch
            if epoch <= 50000:
                from geometry import reduce_matrix_dimensions_by_threshold
                _, V, V_T = reduce_matrix_dimensions_by_threshold(fisher_matrix)
                if epoch == 50000:
                    # Store the projection matrices after the 50000th epoch
                    constant_projection_matrix = V
                    constant_projection_matrix_T = V_T
            else:
                # Use constant projection matrices after the 50000th epoch
                V = constant_projection_matrix
                V_T = constant_projection_matrix_T

            # Compute and store eigenvalues
            eigenvalues = jnp.linalg.eigvalsh(fisher_matrix)
            eigenvalues_history.append(np.array(eigenvalues)) 

            rank = calculate_rank(fisher_matrix)
            rank_history.append(rank)
            epochs_list.append(epoch)

            # Compute Ricci scalar, Kretschmann scalar, and Weyl scalar
            if no_ricci:
                ricci_scalar, kretschmann_scalar, weyl_scalar = 0, 0, 0
            else:
                ricci_scalar, kretschmann_scalar, weyl_scalar = compute_ricci_tensor_from_fisher(params, subsample_x(X, NUMBER_POINTS_USED_FOR_RICCI))
                
            ricci_history.append(ricci_scalar)
            kretschmann_history.append(kretschmann_scalar)
            weyl_history.append(weyl_scalar)
            ricci_epochs.append(epoch)
            
            progress.set_description(f"Epoch {epoch}, Loss: {loss_value:.4f}, Acc: {acc_value:.4f}")

    # Calculate final Fisher matrix and eigenvalues
    fisher_matrix = fisher(params, X)
    eigenvalues = jnp.linalg.eigvalsh(fisher_matrix)

    # Save results
    save_results(output_dir, timestamp, total_params, epochs, lr, 
                loss_history, acc_history, rank_history, eigenvalues,
                ricci_history, epochs_list)

    # Create and save all plots
    plot_comprehensive(
        params, X, y, loss_history, acc_history, ricci_history,
        rank_history, epochs_list, eigenvalues, output_dir, 
        kretschmann_history, weyl_history, size=7
    )
    
    show_all_plots_together(
        loss_history, acc_history, rank_history, epochs_list,
        eigenvalues, ricci_history, np.array(fisher_matrix), output_dir, 
        kretschmann_history, weyl_history
    )
    
    create_eigenvalue_slideshow(epochs_list, eigenvalues_history, output_dir)

def save_results(output_dir, timestamp, total_params, epochs, lr, 
                loss_history, acc_history, rank_history, eigenvalues,
                ricci_history, epochs_list):
    """Save training results to files."""
    
    # Save Ricci scalar values
    ricci_scalar_file = os.path.join(output_dir, "ricci_scalar_values.txt")
    with open(ricci_scalar_file, "w") as f:
        for value in ricci_history:
            f.write(f"{value}\n")
    print(f"Ricci scalar values saved to {ricci_scalar_file}")

    # Save epochs to a text file
    epochs_file = os.path.join(output_dir, "epochs_list.txt")
    with open(epochs_file, "w") as f:
        for epoch in epochs_list:
            f.write(f"{epoch}\n")
    print(f"Epochs saved to {epochs_file}")

    # Save summary info to text
    with open(os.path.join(output_dir, "run_info.txt"), "w") as f:
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Epochs: {epochs}, Learning rate: {lr}\n")
        f.write(f"Final loss: {loss_history[-1]}\n")
        f.write(f"Final accuracy: {acc_history[-1]}\n")
        f.write(f"Final rank: {rank_history[-1]}\n")
        f.write(f"Eigenvalues:\n{np.array(eigenvalues)}\n")
        f.write("\nModel Architecture:\n")
        f.write(f"Number of hidden layers: {len(HIDDEN_SIZES)}\n")
        f.write(f"Activation function: {ACT_FUNCTION}\n")
        f.write(f"Width of each layer: Input=16, Hidden layers={HIDDEN_SIZES}, Output={NUM_CLASSES}\n")  
        f.write(f"Total number of samples used by ricci: {NUMBER_POINTS_USED_FOR_RICCI}\n")
        f.write(f"Number of points for MNIST: {NUMBER_SAMPLES_MNIST}\n")

    # Save eigenvalues to CSV-style file
    np.savetxt(os.path.join(output_dir, "eigenvalues.csv"), np.array(eigenvalues), delimiter=",")

if __name__ == "__main__":
    main()