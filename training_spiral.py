"""
Training module for XOR spiral task with multi-layer neural networks.
"""
import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
import numpy as np
from config import NUM_CLASSES, ACT_FUNCTION, HIDDEN_SIZES

def init_params_10_hidden(key, hidden_sizes):
    key_number = len(hidden_sizes) + 1
    keys = random.split(key, key_number)
    params = {}

    # Initialize weights and biases for each layer
    for i in range(len(hidden_sizes)):
        input_size = 2 if i == 0 else hidden_sizes[i - 1]
        output_size = hidden_sizes[i]
        scale = 1.0 / np.sqrt(input_size)
        params[f"W{i+1}"] = random.normal(keys[i], (input_size, output_size)) * scale
        params[f"b{i+1}"] = random.normal(keys[i], (output_size,)) * scale

    # Output layer (last hidden layer -> 2 classes)
    scale = 1.0 / np.sqrt(hidden_sizes[-1])
    params["W_out"] = random.normal(keys[-1], (hidden_sizes[-1], 2)) * scale
    params["b_out"] = jnp.zeros((2,))

    return params


@jit
def forward(params, x):
    """
    Forward pass for a multi-layer neural network.

    Args:
        params (dict): Dictionary of model parameters (weights and biases).
        x (jax.numpy.ndarray): Input data.

    Returns:
        jax.numpy.ndarray: Output probabilities.
    """
    h = x
    n = len(HIDDEN_SIZES)
    
    for i in range(1, n+1):
        linear = jnp.dot(h, params[f"W{i}"]) + params[f"b{i}"]
        
        if ACT_FUNCTION == "sigmoid":
            h = jax.nn.sigmoid(linear)
        elif ACT_FUNCTION == "tanh":
            h = jax.nn.tanh(linear)
        elif ACT_FUNCTION == "relu":
            h = jax.nn.relu(linear)
        elif ACT_FUNCTION == "elu":
            h = jax.nn.elu(linear)
        elif ACT_FUNCTION == "leaky_relu":
            h = jax.nn.leaky_relu(linear)
        elif ACT_FUNCTION == "swish":
            h = jax.nn.swish(linear)
        elif ACT_FUNCTION == "gelu":
            h = jax.nn.gelu(linear)
        elif ACT_FUNCTION == "softplus":
            h = jax.nn.softplus(linear)
        elif ACT_FUNCTION == "softsign":
            h = jax.nn.softsign(linear)
        elif ACT_FUNCTION == "silu":
            h = jax.nn.silu(linear)
        else:
            raise ValueError(f"Unknown activation function: {ACT_FUNCTION}")
        
    # Output layer with softmax activation
    logits = jnp.dot(h, params["W_out"]) + params["b_out"]
    return jax.nn.softmax(logits)


@jit
def loss_fn(params, x, y):
    """Cross-entropy loss function."""
    preds = forward(params, x)
    return -jnp.mean(jnp.sum(y * jnp.log(preds + 1e-7), axis=1))


@jit
def accuracy(params, x, y):
    """Calculate accuracy."""
    preds = forward(params, x)
    return jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(y, axis=1))


@jit
def train_step(params, x, y, lr):
    """Single training step with gradient descent."""
    grads = grad(loss_fn)(params, x, y)
    return {k: params[k] - lr * grads[k] for k in params}


def train_epoch(params, x_train, y_train, lr, batch_size):
    """
    Train for one epoch with mini-batch gradient descent.
    
    Args:
        params: Model parameters
        x_train: Training inputs
        y_train: Training labels
        lr: Learning rate
        batch_size: Batch size
        
    Returns:
        Updated parameters
    """
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


def count_parameters(params):
    """Count the total number of parameters in the model."""
    total_params = 0
    for key, value in params.items():
        total_params += value.size
    return total_params


def save_initial_parameters(params, output_dir):
    """
    Save the initial parameters of the model to a file.

    Args:
        params (dict): Dictionary of model parameters.
        output_dir (str): Directory to save the file.
    """
    import os
    filename = os.path.join(output_dir, "initial_parameters.txt")
    with open(filename, "w") as f:
        for param_name, param_value in params.items():
            f.write(f"{param_name}:\n{param_value}\n\n")
    print(f"Initial parameters saved to {filename}")