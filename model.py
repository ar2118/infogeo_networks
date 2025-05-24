import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from config import ACT_FUNCTION, HIDDEN_SIZES, NUM_CLASSES 

@jax.jit
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