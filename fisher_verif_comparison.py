import jax
import jax.numpy as jnp
from jax import random, vmap, jacrev
from jax.flatten_util import ravel_pytree
import numpy as np
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
# Set random seed
key = random.PRNGKey(42)

# Simple softmax model (like your neural network's output layer)
def init_softmax_params(key, input_dim=2, num_classes=4):
    return {'W': random.normal(key, (input_dim, num_classes)) * 0.1,
            'b': jnp.zeros(num_classes)}

def forward(params, x):
    logits = jnp.dot(x, params['W']) + params['b']
    return jax.nn.softmax(logits)

# Here's your actual fisher() implementation from the code
def fisher_implementation(params, x):
    flat_params, unravel_fn = ravel_pytree(params)
    num_params = flat_params.shape[0]
    num_classes = 4  # Same as your code
    
    def log_probs_fn(flat_params, x_i, c):
        probs = forward(unravel_fn(flat_params), x_i.reshape(1, -1)).squeeze()
        return jnp.log(probs[c])
    
    def per_sample_fisher(x_i):
        p_i = forward(params, x_i)
        contrib = jnp.zeros((num_params, num_params))
        
        for c in range(num_classes):
            J_c = jacrev(log_probs_fn)(flat_params, x_i, c)
            contrib += p_i[c] * jnp.outer(J_c, J_c)
        return contrib
    
    # Use vmap to compute per-sample Fisher matrices
    fisher_samples = vmap(per_sample_fisher)(x)
    
    # Take the mean to get the average Fisher matrix
    fisher_matrix = jnp.mean(fisher_samples, axis=0)
    return fisher_matrix

# Alternative implementation for verification
def fisher_alternative(params, x):
    flat_params, unravel_fn = ravel_pytree(params)
    num_params = flat_params.shape[0]
    num_classes = 4
    
    # Calculate Fisher by direct summation over samples
    F_total = jnp.zeros((num_params, num_params))
    
    for i in range(x.shape[0]):
        x_i = x[i]
        p_i = forward(params, x_i)
        
        F_sample = jnp.zeros((num_params, num_params))
        for c in range(num_classes):
            # Define a function for the log-probability of class c
            def log_prob_c(params_flat):
                return jnp.log(forward(unravel_fn(params_flat), x_i)[c])
            
            # Get gradient
            grad_c = jacrev(log_prob_c)(flat_params)
            
            # Add to Fisher for this sample
            F_sample += p_i[c] * jnp.outer(grad_c, grad_c)
        
        F_total += F_sample
    
    return F_total / x.shape[0]

# Generate simple test data
n_samples = 100
x_data = random.normal(key, (n_samples, 2))

# Initialize parameters
params = init_softmax_params(key)

# Calculate Fisher matrix using both methods
fisher_result = fisher_implementation(params, x_data)
fisher_alt = fisher_alternative(params, x_data)

# Compare results
print("Fisher Matrix (Your Implementation):")
print(fisher_result)
print("\nFisher Matrix (Alternative Implementation):")
print(fisher_alt)
print("\nDifference:")
print(jnp.abs(fisher_result - fisher_alt))
print("\nRelative Error:")
print(jnp.linalg.norm(fisher_result - fisher_alt) / jnp.linalg.norm(fisher_alt))

# Plot comparison
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(fisher_result, cmap='viridis')
plt.colorbar()
plt.title('Your Fisher Implementation')

plt.subplot(1, 2, 2)
plt.imshow(fisher_alt, cmap='viridis')
plt.colorbar()
plt.title('Alternative Implementation')

plt.tight_layout()
#plt.savefig('nn_fisher_verification.png')
plt.show()