import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time
from datetime import datetime
from tqdm import trange
import os

# Local imports
from config import *
from data_generation_spiral import generate_spiral_data, subsample_x
from training_spiral import (init_params_10_hidden, train_epoch, loss_fn, accuracy, count_parameters, save_initial_parameters)
from visuals_spiral import (plot_comprehensive, show_all_plots_together, create_eigenvalue_slideshow)
from geometry import (fisher_reduced, calculate_rank, compute_ricci_tensor_from_fisher)

# Configure JAX
print(NO_RICCI)
jax.config.update("jax_enable_x64", True)

def main(number_epoch = NUMBER_EPOCH, learning_rate = LEARNING_RATE, many_layers=True, key_manual_check=True, no_ricci=NO_RICCI):

    # Generate data
    X, y, centers, colors = generate_spiral_data(seed=42)
    X_val, y_val, _, _ = generate_spiral_data(seed=42)


    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"plot_summer_removed_params_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

        # Initialize parameters
    if key_manual_check:
        key = random.PRNGKey(KEY_MANUAL)
    else:
        key = random.PRNGKey(int(time.time()))  # Use the current time as the seed
    
    params = init_params_10_hidden(key, HIDDEN_SIZES)

    save_initial_parameters(params, output_dir)

    total_params = count_parameters(params)
    print(f"Total parameters: {total_params}")    
    # Training hyperparameters
    epochs =  number_epoch# Reduced from 100000 to make it run faster
    lr = learning_rate

    loss_history = []
    acc_history = []
    rank_history = []
    epochs_list = []
    acc_val_history = []  # <-- Add this for validation accuracy
    ricci_history = []
    ricci_epochs = []
    eigenvalues_history = []
    kretschmann_history = []
    weyl_history = []


    # Training loop
    progress = trange(epochs, desc="Training", leave=True)
    for epoch in progress:
        params = train_epoch(params, X, y, lr, batch_size=80)
        
        # Record metrics every 50 epochs
        if epoch > 50 and epoch % 500 == 0 or epoch == epochs - 1:
            loss = loss_fn(params, X, y)
            acc = accuracy(params, X, y)
            acc_val = accuracy(params, X_val, y_val)  # Validation accuracy
            loss_history.append(loss)
            acc_history.append(acc)
            acc_val_history.append(acc_val)  # Store validation accuracy
            fisher_matrix = fisher_reduced(params, X)
            eigenvalues = jnp.linalg.eigvalsh(fisher_matrix)
            rank = calculate_rank(fisher_matrix)

            # Compute and store eigenvalues
            eigenvalues = jnp.linalg.eigvalsh(fisher_matrix)
            eigenvalues_history.append(np.array(eigenvalues)) 

            rank = calculate_rank(fisher_matrix)
            rank_history.append(rank)
            epochs_list.append(epoch)

            # Compute Ricci scalar, Kretschmann scalar, and Weyl scalar
            if no_ricci:
                ricci_scalar, kretschmann_scalar, weyl_scalar, FISHER_USED = 0, 0, 0, []
            else:
                ricci_scalar, kretschmann_scalar, weyl_scalar, FISHER_USED = compute_ricci_tensor_from_fisher(params, subsample_x(X, NUMBER_POINTS_USED_FOR_RICCI))
            ricci_history.append(ricci_scalar)
            kretschmann_history.append(kretschmann_scalar)
            weyl_history.append(weyl_scalar)
            ricci_epochs.append(epoch)
            
            progress.set_description(f"Epoch {epoch}, Acc: {acc:.4f}")

    # Calculate Fisher matrix and eigenvalues
    fisher_matrix = fisher_reduced(params, X)
    eigenvalues = jnp.linalg.eigvalsh(fisher_matrix)

    # Save epochs to a text file
    epochs_file = os.path.join(output_dir, f"epochs_list_{timestamp}.txt")
    with open(epochs_file, "w") as f:
        for epoch in epochs_list:
            f.write(f"{epoch}\n")
    print(f"Epochs saved to {epochs_file}")

    # Save epochs to a text file
    ricci_file = os.path.join(output_dir, f"ricci_list_{timestamp}.txt")
    with open(ricci_file, "w") as f:
        for ricci_value in ricci_history:
            f.write(f"{ricci_value}\n")
    print(f"Epochs saved to {ricci_file}")

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
        f.write(f"Number of hidden layers: 1\n")  # Single hidden layer
        f.write(f"Activation function: tanh\n")  # Activation function used
        f.write(f"Width of each layer: Input=16, Output={NUM_CLASSES}\n")  
        f.write(f"Total number of samples used by ricci: {NUMBER_POINTS_USED_FOR_RICCI}\n")
        f.write(f"Number of points on the spiral: {NUMBER_POINTS_ON_SPIRAL}\n")
        f.write(f"hidden sizes: {HIDDEN_SIZES}\n")
        f.write(f"activation function: {ACT_FUNCTION}\n")

    # Save eigenvalues to CSV-style file
    np.savetxt(os.path.join(output_dir, "fishermatrix.csv"), np.array(fisher_matrix), delimiter=",")
    np.savetxt(os.path.join(output_dir, "fishermatrix_used.csv"), np.array(FISHER_USED), delimiter=",")

    np.savetxt(os.path.join(output_dir, "eigenvalues.csv"), np.array(eigenvalues), delimiter=",")

    show_all_plots_together(loss_history, acc_history, rank_history, epochs_list,
    eigenvalues, ricci_history, np.array(fisher_matrix), output_dir, kretschmann_history, weyl_history, acc_val_history=acc_val_history)
    #plt.show()

    #plot_eigenvalues_over_time(epochs_list, eigenvalues_history, output_dir)

    #create_eigenvalue_slideshow(epochs_list, eigenvalues_history, output_dir)
    #plt.show()

if __name__ == "__main__":
    main()