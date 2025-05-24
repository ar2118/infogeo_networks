import sys
if sys.platform == "win32":
    import types
    sys.modules['resource'] = types.ModuleType('resource')  # Fake empty module
import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap, jacrev
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from jax.flatten_util import ravel_pytree
import time
from datetime import datetime
from jax import jacfwd, jacrev
from jax import tree_util
import os
import tensorflow_datasets as tfds
import tensorflow as tf
import scienceplots 
plt.style.use(['science', 'no-latex'])
from jax import jacobian
from jax.numpy.linalg import matrix_rank
jax.config.update("jax_enable_x64", True)

rank_tol = 1e-15 # Default tolerance, matches the one used in plot_results
num_classes = 2
NUMBER_EPOCH = 20000
LEARNING_RATE = 0.006
NOMBRE_POINTS_SUR_SPIRALE = 0
NOMBRE_POINTS_UTILISES_PAR_RICCI = 50
NUM_SAMPLES_MNIST = 50
act_function = 'tanh'  # Activation function, options: "tanh", "relu", "sigmoid", "elu", "softmax", "swish"
hidden_sizes = [2]  # Example sizes for 10 layers
key_manual = 23

def main(number_epoch = NUMBER_EPOCH, learning_rate = LEARNING_RATE, key_manual_check = False):

    # Generate data
    X, y = load_mnist_8_9()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"MNIST4_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize parameters
    if key_manual_check:
        key = random.PRNGKey(key_manual)
    else:
        key = random.PRNGKey(int(time.time()))  # Use the current time as the seed
    '''
    W1 = jnp.array([[1.32938189, -0.34027037],
                    [0.29377262,  1.68421527]])
    b1 = jnp.array([1.32938189, -0.34027037])
    W2 = jnp.array([[-0.99057465,  1.01267944],
                    [0.44180788,  0.14176591]])
    b2 = jnp.array([-0.99057465,  1.01267944])
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    '''
    params = init_params_10_hidden(key, hidden_sizes=hidden_sizes)

    total_params = count_parameters(params)
    print(f"Total parameters: {total_params}")    
    # Training hyperparameters
    epochs =  number_epoch# Reduced from 100000 to make it run faster
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
    ricci_scalar_5000_5050 = []
    ricci_scalar_5000_5050_epochs = []
    # Initialize constant projection matrices
    constant_projection_matrix = None
    constant_projection_matrix_T = None
    
    # Training loop
    progress = trange(epochs, desc="Training", leave=True)
    for epoch in progress:
        params = train_epoch(params, X, y, lr, batch_size=32)
        
        # Record metrics every 50 epochs
        if epoch > 50 and epoch % 100 == 0 or epoch == epochs - 1:
            loss = loss_fn(params, X, y)
            acc = accuracy(params, X, y)
            
            # Convert JAX arrays to Python floats
            loss_value = float(loss)
            acc_value = float(acc)
            
            loss_history.append(loss_value)
            acc_history.append(acc_value)
            
            fisher_matrix = fisher(params, X)

            # Compute projection matrices dynamically before the 1000th epoch
            if epoch <= 50000:
                _, V, V_T = reduce_matrix_dimensions_by_threshold(fisher_matrix)
                if epoch == 50000:
                    # Store the projection matrices after the 1000th epoch
                    constant_projection_matrix = V
                    constant_projection_matrix_T = V_T
            else:
                # Use constant projection matrices after the 1000th epoch
                V = constant_projection_matrix
                V_T = constant_projection_matrix_T

            # Compute and store eigenvalues
            eigenvalues = jnp.linalg.eigvalsh(fisher_matrix)
            eigenvalues_history.append(np.array(eigenvalues)) 

            rank = calculate_rank(fisher_matrix)
            rank_history.append(rank)
            epochs_list.append(epoch)

            # Compute Ricci scalar, Kretschmann scalar, and Weyl scalar
            ricci_scalar, kretschmann_scalar, weyl_scalar = compute_ricci_tensor_from_fisher(params, subsample_x(X))
            ricci_history.append(ricci_scalar)
            kretschmann_history.append(kretschmann_scalar)
            weyl_history.append(weyl_scalar)
            ricci_epochs.append(epoch)
            
            progress.set_description(f"Epoch {epoch}, Loss: {loss_value:.4f}, Acc: {acc_value:.4f}")

    # Calculate Fisher matrix and eigenvalues
    fisher_matrix = fisher(params, X)
    eigenvalues = jnp.linalg.eigvalsh(fisher_matrix)

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
        f.write(f"Number of hidden layers: 1\n")  # Single hidden layer
        f.write(f"Activation function: tanh\n")  # Activation function used
        f.write(f"Width of each layer: Input=16, Hidden layers={hidden_sizes}, Output={num_classes}\n")  
        f.write(f"Total number of samples used by ricci: {NOMBRE_POINTS_UTILISES_PAR_RICCI}\n")
        f.write(f"Number of points on the spiral: {NOMBRE_POINTS_SUR_SPIRALE}\n")


    # Save eigenvalues to CSV-style file
    np.savetxt(os.path.join(output_dir, "eigenvalues.csv"), np.array(eigenvalues), delimiter=",")

    print(ricci_scalar_5000_5050)
    print(ricci_scalar_5000_5050_epochs)
    # Create and save all plots to output_dir
    plot_comprehensive(
        params, X, y,
        loss_history, acc_history, ricci_history,
        rank_history, epochs_list, eigenvalues,
        output_dir, kretschmann_history, weyl_history, size=7
    )
    plot_comprehensive(
        params, X, y, 
        loss_history, acc_history, ricci_history,
        rank_history, epochs_list, eigenvalues,
        output_dir, kretschmann_history, weyl_history, size=5
    )
    show_all_plots_together(loss_history, acc_history, rank_history, epochs_list,
    eigenvalues, ricci_history, np.array(fisher_matrix), output_dir, kretschmann_history, weyl_history)
    #plt.show()

    #plot_eigenvalues_over_time(epochs_list, eigenvalues_history, output_dir)

    create_eigenvalue_slideshow(epochs_list, eigenvalues_history, output_dir)
    #plt.show()

# Dérivé de "test17XOR Avec print rang utilisé le 02 04 2025.py"
# Generate 5 Gaussian clusters with XOR pattern
def load_mnist_8_9(split='train', num_samples=NUM_SAMPLES_MNIST, seed=0):
    ds, _ = tfds.load("mnist", split=split, as_supervised=True, with_info=True)
    print(f"Dataset loaded: {split} split")
    ds = ds.shuffle(10_000, seed=seed)

    X_list, y_list = [], []
    counts = {8: 0, 9: 0}  # To track how many of each digit we've collected
    
    for img, label in tfds.as_numpy(ds):
        # Only process if the label is 8 or 9
        if label not in [8, 9]:
            continue
            
        # Check if we've collected enough samples of this digit
        if counts[label] >= num_samples:
            # If we have enough of both classes, break
            if all(count >= num_samples for count in counts.values()):
                break
            continue
            
        # Process the image
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        img = img.reshape(28, 28)  # Original shape
        
        # Downsample to 5×5 using bilinear interpolation
        img_small = jax.image.resize(img, shape=(4, 4), method="bilinear")
        
        # Create one-hot encoding (2 classes: 8->0, 9->1)
        y_onehot = np.zeros(2, dtype=np.float32)
        y_onehot[label - 8] = 1.0  # Map 8->0, 9->1
        
        X_list.append(img_small.flatten())  # flatten to (25,)
        y_list.append(y_onehot)
        
        counts[label] += 1

    X = jnp.array(X_list)
    y = jnp.array(y_list)
    
    print(f"Loaded {len(X)} samples with shape {X.shape}")
    print(f"Class distribution: {counts}")
    
    return X, y

    k1, k2, k3 = random.split(key, 3)
    # Scale for initialization
    scale1 = 1.0 / np.sqrt(16)  # Input dimension is 2
    scale2 = 1.0 / np.sqrt(hidden_size1)
    scale3 = 1.0 / np.sqrt(hidden_size2)

    # First hidden layer (2 -> hidden_size1)
    W1 = random.normal(k1, (16, hidden_size1)) * scale1  
    b1 = jnp.zeros((hidden_size1,))

    # Second hidden layer (hidden_size1 -> hidden_size2)
    W2 = random.normal(k2, (hidden_size1, hidden_size2)) * scale2
    b2 = jnp.zeros((hidden_size2,))

    # Output layer (hidden_size -> 2) - Binary classification
    W3 = random.normal(k3, (hidden_size2, 2)) * scale3
    b3 = jnp.zeros((2,))

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

'''
def init_params_old(key, hidden_size=hidden_size):
    k1, k2 = random.split(key, 2)
    # Scale for initialization
    scale1 = 1.0 / np.sqrt(16)  # Input dimension is 2
    scale2 = 1.0 / np.sqrt(hidden_size)
    
    # Hidden layer (2 -> hidden_size)
    W1 = random.normal(k1, (16, hidden_size)) * scale1  
    b1 = random.normal(k1, (hidden_size,)) * scale1  # Bias initialized with the same key as W1
    
    # Output layer (hidden_size -> 2) - Binary classification
    W2 = random.normal(k2, (hidden_size, 2)) * scale2
    b2 = random.normal(k2, (2,)) * scale2  # Bias initialized with the same key as W2

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
'''

def init_params_10_hidden(key, hidden_sizes=hidden_sizes):
    """
    Initialize parameters for a 10-layer neural network.

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        hidden_sizes (list): List of integers specifying the size of each hidden layer.

    Returns:
        dict: Dictionary containing weights and biases for each layer.
    """
    key_number = len(hidden_sizes) + 1
    keys = random.split(key, key_number)  # 10 hidden layers + 1 output layer
    params = {}

    # Initialize weights and biases for each layer
    for i in range(len(hidden_sizes)):
        input_size = 16 if i == 0 else hidden_sizes[i - 1]
        output_size = hidden_sizes[i]
        scale = 1.0 / np.sqrt(input_size)
        params[f"W{i+1}"] = random.normal(keys[i], (input_size, output_size)) * scale
        params[f"b{i+1}"] = jnp.zeros((output_size,))

    # Output layer (last hidden layer -> 2 classes)
    scale = 1.0 / np.sqrt(hidden_sizes[-1])
    params["W_out"] = random.normal(keys[-1], (hidden_sizes[-1], 2)) * scale
    params["b_out"] = jnp.zeros((2,))

    return params

@jit
def forward(params, x):
    """
    Forward pass for a 10-layer neural network.

    Args:
        params (dict): Dictionary of model parameters (weights and biases).
        x (jax.numpy.ndarray): Input data.

    Returns:
        jax.numpy.ndarray: Output probabilities.
    """
    h = x
    n = len(hidden_sizes)
    for i in range(1, n+1):  # Loop through the 10 hidden layers
        if act_function == 'tanh':
            h = jax.nn.tanh(jnp.dot(h, params[f"W{i}"]) + params[f"b{i}"])
        elif act_function == 'relu':
            h = jax.nn.relu(jnp.dot(h, params[f"W{i}"]) + params[f"b{i}"])
        elif act_function == 'sigmoid':
            h = jax.nn.sigmoid(jnp.dot(h, params[f"W{i}"]) + params[f"b{i}"])
        elif act_function == 'elu':
            h = jax.nn.elu(jnp.dot(h, params[f"W{i}"]) + params[f"b{i}"])   
        elif act_function == 'softmax':
            h = jax.nn.softmax(jnp.dot(h, params[f"W{i}"]) + params[f"b{i}"])
        elif act_function == 'swish':
            h = jax.nn.swish(jnp.dot(h, params[f"W{i}"]) + params[f"b{i}"])
        elif act_function == 'leaky_relu':
            h = jax.nn.leaky_relu(jnp.dot(h, params[f"W{i}"]) + params[f"b{i}"])

    # Output layer with softmax activation
    logits = jnp.dot(h, params["W_out"]) + params["b_out"]
    return jax.nn.softmax(logits)  # Returns probabilities for 2 classes


@jit
def forward_old(params, x):
    # Hidden layer with sigmoid activation
    h = jax.nn.tanh(jnp.dot(x, params['W1']) + params['b1'])
    # Output layer with softmax activation
    logits = jnp.dot(h, params['W2']) + params['b2']
    return jax.nn.softmax(logits)  # Returns probabilities for 4 classes

@jit
def loss_fn(params, x, y):
    preds = forward(params, x)
    return -jnp.mean(jnp.sum(y * jnp.log(preds + 1e-7), axis=1))

@jit
def accuracy(params, x, y):
    preds = forward(params, x)
    return jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(y, axis=1))

@jit
def train_step(params, x, y, lr):
    grads = grad(loss_fn)(params, x, y)
    return {k: params[k] - lr * grads[k] for k in params}

def train_epoch(params, x_train, y_train, lr, batch_size):
    num_samples = x_train.shape[0]
    num_batches = num_samples // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        x_batch = x_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        params = train_step(params, x_batch, y_batch, lr)
    
    # Handle remaining samples
    if num_samples % batch_size != 0:
        x_batch = x_train[num_batches * batch_size:]
        y_batch = y_train[num_batches * batch_size:]
        params = train_step(params, x_batch, y_batch, lr)
    
    return params

def calculate_rank(matrix, tol=rank_tol):
    """Calculate the rank of a matrix based on SVD and a threshold."""
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    rank = jnp.sum(singular_values > tol)
    return rank

# Save eigenvalues to a file
def save_eigenvalues_to_file(matrix, filename="eigenvalues.txt"):
    """Save eigenvalues of a matrix to a file."""
    # Convert the matrix to a NumPy array (if it's a JAX array)
    matrix_np = np.array(matrix)

    # Compute the eigenvalues
    eigenvalues = np.linalg.eigvals(matrix_np)

    # Save the eigenvalues to a file
    np.savetxt(filename, eigenvalues, delimiter=",")
    print(f"Eigenvalues saved to {filename}")
    return eigenvalues

# Count parameters in a model
def count_parameters(params):
    """Count the number of parameters in the model."""
    total_params = 0
    for key, value in params.items():
        total_params += value.size
    return total_params

def fisher(params, x):
    flat_params, unravel_fn = ravel_pytree(params)
    num_params = flat_params.shape[0]
    
    def log_probs_fn(flat_params, x_i, c):
        probs = forward(unravel_fn(flat_params), x_i.reshape(1, -1)).squeeze()
        return jnp.log(probs[c])
    
    def per_sample_fisher(flat_params, x_i):
        p_i = forward(unravel_fn(flat_params), x_i.reshape(1, -1)).squeeze()
        contrib = jnp.zeros((num_params, num_params))
        
        for c in range(num_classes):
            J_c = jacrev(log_probs_fn)(flat_params, x_i, c)
            contrib += p_i[c] * jnp.outer(J_c, J_c)
        return contrib
    
    fisher_samples = vmap(per_sample_fisher, in_axes=(None, 0))(flat_params, x)
    fisher_matrix = jnp.mean(fisher_samples, axis=0)

    return fisher_matrix

def reduce_matrix_dimensions_by_threshold(matrix, threshold=rank_tol):

    # Convert to numpy if it's a JAX array
    if hasattr(matrix, 'device_buffer'):
        matrix_np = np.array(matrix)
    else:
        matrix_np = matrix
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix_np)

    # Find eigenvalues above the threshold
    mask = eigenvalues > threshold
    kept_indices = np.where(mask)[0]

    # Calculate the rank of the matrix
    rank = calculate_rank(matrix, tol=rank_tol)
    
    # Sort eigenvalues in descending order and keep the indices
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Find eigenvalues above the threshold
    mask = eigenvalues > threshold
    kept_indices = np.where(mask)[0]
    
    # If no eigenvalues are above threshold, keep at least one
    if len(kept_indices) == 0:
        kept_indices = [0]
    
    # Keep only the eigenvectors with eigenvalues >= threshold
    V = eigenvectors[:, kept_indices]  # Shape: (n, r) where r is the number of kept eigenvectors
    
    # Transpose of V for convenience
    V_T = V.T  # Shape: (r, n)

    # Project the original matrix to the reduced space: V^T * A * V
    reduced_matrix = V_T @ matrix_np @ V  # M×M matrix
    
    return reduced_matrix, V, V_T

def compute_ricci_tensor_from_fisher(params, x, constant_V=None, constant_V_T=None):
    flat_params, unravel_fn = ravel_pytree(params)

    # ---- Metric: Fisher matrix ---- 
    # This extra internal function is necessary to get the Fisher via flattened params
    @jit
    def fisher_internal(flat_params, x_data):
        num_params = flat_params.shape[0]
        
        def log_probs_fn(flat_params, x_i, c):
            params_dict = unravel_fn(flat_params)
            probs = forward(params_dict, x_i.reshape(1, -1)).squeeze()
            return jnp.log(probs[c])
        
        def per_sample_fisher(flat_params, x_i):
            params_dict = unravel_fn(flat_params)
            p_i = forward(params_dict, x_i.reshape(1, -1)).squeeze()
            contrib = jnp.zeros((num_params, num_params))
            
            for c in range(num_classes):
                J_c = jacrev(log_probs_fn)(flat_params, x_i, c)
                contrib += p_i[c] * jnp.outer(J_c, J_c)
            return contrib
        
        fisher_samples = vmap(per_sample_fisher, in_axes=(None, 0))(flat_params, x_data)
        fisher_matrix = jnp.mean(fisher_samples, axis=0)

        return fisher_matrix

    # ---- Reduced Space Fisher matrix ----
    @jit
    def fisher_reduced(reduced_params, x_data): #change so that it does the projection internally
        
        # Get the full parameters from the reduced ones
        full_params = projection_matrix @ reduced_params
        
        # Use the same function as before to compute Fisher on full params
        def log_probs_fn_reduced(full_p, x_i, c):
            params_dict = unravel_fn(full_p)
            probs = forward(params_dict, x_i.reshape(1, -1)).squeeze()
            return jnp.log(probs[c])
            
        def per_sample_fisher_red(full_p, x_i):
            p_i = forward(unravel_fn(full_p), x_i.reshape(1, -1)).squeeze()
            contrib = jnp.zeros((full_p.shape[0], full_p.shape[0]))
            for c in range(num_classes):
                J_c = jacrev(log_probs_fn_reduced)(full_p, x_i, c)
                contrib += p_i[c] * jnp.outer(J_c, J_c)
            return contrib
            
        # Compute Fisher in full space
        fisher_samples_red = vmap(per_sample_fisher_red, in_axes=(None, 0))(full_params, x_data)
        full_fisher = jnp.mean(fisher_samples_red, axis=0)
        
        # Project back to reduced space: V_T * G * V
        return projection_matrix_T @ full_fisher @ projection_matrix

    # ---- Christoffel symbols (i,j,k) ----
    @jit
    def christoffel_symbols_NEW(dg, g_inv):
        partial_derivs = jnp.einsum('mns -> smn', dg)
        sum_partial_derivs = partial_derivs + jnp.einsum('nrm -> mnr', partial_derivs) - jnp.einsum('rmn -> mnr', partial_derivs)
        christ = 0.5 * jnp.einsum('sr, mnr -> smn', g_inv, sum_partial_derivs)
        return christ

    def riemann_curvature(Gamma, dGamma):
        """
        Compute the Riemann curvature at a given coordinate.
        """
        dGamma = jnp.einsum('rmns -> srmn', dGamma) # Rearranging indices because when we differentiate we get the extra index at the END
        term1 = jnp.einsum('mrns -> rsmn', dGamma)
        term2 = jnp.einsum('nrms -> rsmn', dGamma)
        term3 = jnp.einsum('rml, lns -> rsmn', Gamma, Gamma)
        term4 = jnp.einsum('rnl, lms -> rsmn', Gamma, Gamma)
        return term1 - term2 + term3 - term4
    
    def ricci_tensor(Gamma, dGamma):
        """
        Compute the Ricci tensor at a given coordinate.
        """
        riemann = riemann_curvature(Gamma, dGamma)
        return jnp.einsum('rsru -> su', riemann)

    # ---- Ricci tensor (i,j) ----
    @jit
    def ricci_tensor_fast(Gamma, dGamma):
        dGamma = jnp.einsum('rmns -> srmn', dGamma) # Rearranging indices because when we differentiate we get the extra index at the END
        term1 = jnp.einsum('mrns -> ns', dGamma)               
        term2 = jnp.einsum('nrms -> ns', dGamma)                
        term3 = jnp.einsum('rml, lns -> ns', Gamma, Gamma)              
        term4 = jnp.einsum('rnl, lms -> ns', Gamma, Gamma)      
        return term1 - term2 + term3 - term4

    def kretschmann_scalar(g_inv, R):
        """
        Compute the Kretschmann scalar: K = R_{ijkl} R^{ijkl}
        
        Args:
            g: Metric tensor, shape (n, n)
            g_inv: Inverse metric, shape (n, n)
            R: Riemann tensor, shape (n, n, n, n)

        Returns:
            Kretschmann scalar (float)
        """
        # Raise all indices on Riemann using g_inv
        R_up = jnp.einsum('im,jn,kp,lq,mnpq->ijkl',
                        g_inv, g_inv, g_inv, g_inv, R)

        # Contract with original Riemann to compute scalar
        K = jnp.einsum('ijkl,ijkl->', R, R_up)

        return K
    
    def weyl_tensor(g, riemann, ricci_tensor, ricci_scalar):

        n = g.shape[0]

        # First term: Riemann itself
        C = riemann

        # Second term (with Ricci tensor and metric)
        term2 = (1 / (n - 2)) * (
            jnp.einsum('im,kl->iklm', ricci_tensor, g)
        - jnp.einsum('il,km->iklm', ricci_tensor, g)
        + jnp.einsum('kl,im->iklm', ricci_tensor, g)
        - jnp.einsum('km,il->iklm', ricci_tensor, g)
        )

        # Third term (with Ricci scalar and metric)
        term3 = (ricci_scalar / ((n - 1) * (n - 2))) * (
            jnp.einsum('il,km->iklm', g, g)
        - jnp.einsum('im,kl->iklm', g, g)
        )

        return C + term2 + term3

    def weyl_scalar(g_inv, C):
        # Raise all indices on C_{ijkl} to get C^{ijkl}
        C_up = jnp.einsum('im,jn,kp,lq,mnpq->ijkl',
                        g_inv, g_inv, g_inv, g_inv, C)

        # Contract C_{ijkl} with C^{ijkl}
        C2 = jnp.einsum('ijkl,ijkl->', C, C_up)
        return C2

    
    # Step 2: Reduce dimensionality of the Fisher matrix
    fisher_np = np.array(fisher_internal(flat_params, x))

    # Use constant projection matrices if provided
    if constant_V is not None and constant_V_T is not None:
        V = constant_V
        V_T = constant_V_T
        reduced_matrix = V_T @ fisher_np @ V
    else:
        reduced_matrix, V, V_T = reduce_matrix_dimensions_by_threshold(fisher_np) 

    # Convert reduced matrix back to JAX array
    reduced_fisher = jnp.array(reduced_matrix)
    projection_matrix_T = jnp.array(V_T)  # N×M
    projection_matrix = jnp.array(V)  # M×N

    # Now we have a function which takes the reduced params, rebuilds the fisher and the projects it in reduced space
    reduced_params0 = projection_matrix_T @ flat_params
    g = reduced_fisher  # Already computed above
    g=(g + g.T) / 2  # Ensure symmetry
    g_inv = jnp.linalg.inv(g)
    g_inv = (g_inv + g_inv.T) / 2  # Ensure symmetry
    dg = jacrev(fisher_reduced)(reduced_params0, x)  # Shape: (d, d, d)
    Gamma = christoffel_symbols_NEW(dg, g_inv)

    @jit
    def christoffel_fn(reduced_p, x):
        g_x = fisher_reduced(reduced_p, x)
        g_inv_x = jnp.linalg.inv(g_x)
        dg_x = jacrev(fisher_reduced)(reduced_p, x)    # (d, d, d): ∂g_ij / ∂θ_k
        return christoffel_symbols_NEW(dg_x, g_inv_x)

    dGamma = jacrev(christoffel_fn)(reduced_params0, x)
    ricci = ricci_tensor(Gamma, dGamma)
    ricci = (ricci + ricci.T) / 2  # Ensure symmetry
    ricci_scalar = jnp.einsum('ij,ji->', g_inv, ricci)
    
    riemann = 0#riemann_curvature(Gamma, dGamma)
    kretschmann_scalar_value = 0 #kretschmann_scalar(g_inv, riemann)
    C = 0#weyl_tensor(g, riemann, ricci, ricci_scalar)
    weyl_scalar_value = 0#weyl_scalar(g_inv, C)
    
    return ricci_scalar, kretschmann_scalar_value, weyl_scalar_value

def subsample_x(x, max_points=NOMBRE_POINTS_UTILISES_PAR_RICCI):
    return x[:max_points]

# Save initial parameters to a file
def save_initial_parameters(params, output_dir):
    """
    Save the initial parameters of the model to a file.

    Args:
        params (dict): Dictionary of model parameters.
        output_dir (str): Directory to save the file.
    """
    filename = os.path.join(output_dir, "initial_parameters.txt")
    with open(filename, "w") as f:
        for param_name, param_value in params.items():
            f.write(f"{param_name}:\n{param_value}\n\n")
    print(f"Initial parameters saved to {filename}")

def plot_comprehensive(params, X, y, loss_history, acc_history, ricci_history, 
                      rank_history, epochs_list, eigenvalues, output_dir, kretschmann_history, weyl_history, size=7, **kwargs):
    """Generate and save individual square plots at 500 dpi in PDF format."""
    
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    from matplotlib.ticker import FuncFormatter
    
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
                     hline=(rank_tol, 'r', f'Tol = {rank_tol:.0e}'), dots=True)

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
                     custom_ticks=True  # Enable the custom ticks for this plot
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
    ax4.axhline(rank_tol, color='r', linestyle='--', label=f'Tol={rank_tol:.0e}')
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

if __name__ == "__main__":
    main()