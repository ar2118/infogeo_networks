import jax
import jax.numpy as jnp
from jax import random, vmap, jacrev, jacfwd
from jax.flatten_util import ravel_pytree
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scienceplots 
plt.style.use(['science', 'no-latex'])
jax.config.update("jax_enable_x64", True)


#################################################################
# PART 1: Categorical Distribution Setup
#################################################################
n_samples = 1000
def sample_categorical(key, params, n_samples=n_samples):
    """
    Sample from a categorical distribution with given probabilities.
    
    Args:
        key: JAX random key
        params: Dictionary with 'probs' key containing probabilities
        n_samples: Number of samples to generate
    
    Returns:
        One-hot encoded samples
    """
    probs = params['probs']
    categories = probs.shape[0]
    
    # Generate random samples
    keys = random.split(key, n_samples)
    samples = jnp.array([random.choice(keys[i], categories, p=probs) for i in range(n_samples)])
    # One-hot encode the samples
    one_hot = jnp.zeros((n_samples, categories))
    one_hot = one_hot.at[jnp.arange(n_samples), samples].set(1)
    
    return one_hot

def log_likelihood_categorical(params, x):
    """
    Compute the log-likelihood of a categorical distribution.
    
    Args:
        params: Dictionary with 'probs' key containing probabilities
        x: One-hot encoded observation(s)
    
    Returns:
        Log-likelihood value(s)
    """
    probs = params['probs']
    
    # For one-hot encoded x, this selects the probability of the observed category
    log_probs = jnp.log(probs)
    
    # If x is a single observation
    if x.ndim == 1:
        return jnp.sum(x * log_probs)
    # If x is a batch of observations
    else:
        return jnp.sum(x * log_probs, axis=1)

#################################################################
# PART 2: Fisher Matrix Implementations
#################################################################

def analytical_fisher_categorical(params):
    """
    Calculate the analytical Fisher Information Matrix for a categorical distribution.
    
    For a categorical distribution with k categories and probabilities θᵢ,
    the Fisher Information Matrix is:
    
    F_{ij} = δ_{ij}/θᵢ
    
    Where δ_{ij} is the Kronecker delta.
    
    Args:
        params: Dictionary with 'probs' key containing probabilities
        
    Returns:
        The analytical Fisher Information Matrix
    """
    probs = params['probs']
    categories = probs.shape[0]
    
    # Initialize Fisher matrix
    fisher_matrix = jnp.zeros((categories, categories))
    
    # The diagonal elements are 1/θᵢ
    for i in range(categories):
        fisher_matrix = fisher_matrix.at[i, i].set(1.0 / probs[i])
    
    return fisher_matrix

def your_fisher_implementation(params, x=None):
    """
    Compute the Fisher Information Matrix for a categorical distribution using autodiff.
    
    For a categorical distribution, we calculate:
    F = E_x~p(x|θ)[∇log p(x|θ) ∇log p(x|θ)^T]
    
    We can compute this either:
    1. Exactly by summing over all possible outcomes weighted by their probabilities (if x=None)
    2. Empirically by averaging over a batch of samples (if x is provided)
    
    Args:
        params: Dictionary with 'probs' key containing probabilities
        x: Optional batch of samples as one-hot vectors. If None, calculate exact Fisher.
        
    Returns:
        Fisher Information Matrix
    """
    probs = params['probs']
    categories = probs.shape[0]
    
    # Flatten parameters for autodiff
    flat_params, unravel_fn = ravel_pytree(params)
    num_params = flat_params.shape[0]
    
    # Define forward function for categorical distribution
    def categorical_forward(params, x_i):
        """Return probabilities for a categorical distribution"""
        return params['probs']
    
    def log_probs_fn(flat_params, x_i, c):
        """Compute log probability for a specific category c"""
        p = unravel_fn(flat_params)
        category_probs = categorical_forward(p, x_i)
        return jnp.log(category_probs[c])
    
    def per_sample_fisher(flat_params, x_i):
        """Compute Fisher contribution for a specific sample"""
        # Get probabilities (for categorical, these are the params directly)
        p = unravel_fn(flat_params)
        p_i = categorical_forward(p, x_i)
        
        # Initialize contribution matrix
        contrib = jnp.zeros((num_params, num_params))
        
        # Sum over all categories
        for c in range(categories):
            # Compute Jacobian (gradient) of log probability for category c
            J_c = jacfwd(log_probs_fn)(flat_params, x_i, c)
            
            # Weight by probability of this category
            contrib += p_i[c] * jnp.outer(J_c, J_c)
            
        return contrib

    # Calculate empirical Fisher using provided samples
    fisher_samples = vmap(per_sample_fisher, in_axes=(None, 0))(flat_params, x)
    
    # Simply average over samples
    fisher_matrix = jnp.mean(fisher_samples, axis=0)
    
    return fisher_matrix

#################################################################
# PART 3: Verification
#################################################################

def verify_fisher_categorical():
    print("Verifying Fisher Information Matrix calculation for categorical distribution...")
    
    # Set up a 4D categorical distribution
    categories = 6
    key = random.PRNGKey(42)
    
    # Generate random probabilities that sum to 1
    key, subkey = random.split(key)
    raw_probs = random.uniform(subkey, shape=(categories,), minval=0.05, maxval=1.0)
    probs = raw_probs / jnp.sum(raw_probs)
    
    params = {'probs': probs}
    X = sample_categorical(key, params)
    print(X)
    print(f"\nProbabilities: {probs}")
    
    # Calculate analytical Fisher
    analytical_fisher = analytical_fisher_categorical(params)
    
    # Calculate Fisher using your implementation
    your_fisher = your_fisher_implementation(params, X)
    
    # Calculate error metrics
    abs_diff = jnp.abs(your_fisher - analytical_fisher)
    rel_error = jnp.linalg.norm(abs_diff) / jnp.linalg.norm(analytical_fisher)
    
    print("\nAnalytical Fisher:")
    print(analytical_fisher)
    print("\nFisher Implementation:")
    print(your_fisher)
    print(f"\nRelative Error: {rel_error:.16f}")
    
    # Visualize the comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(analytical_fisher, cmap='viridis')
    plt.colorbar()
    plt.title('Analytical Fisher (Diagonal Matrix)')
    
    plt.subplot(1, 3, 2)
    plt.imshow(your_fisher, cmap='viridis')
    plt.colorbar()
    plt.title('Fisher Implementation')
    
    plt.subplot(1, 3, 3)
    plt.imshow(abs_diff, cmap='inferno')
    plt.colorbar()
    plt.title(f'Absolute Difference (Rel Error: {rel_error:.1e})')
    # Add a text annotation with the relative error in scientific notation
    plt.figtext(0.5, 0.05, f"Overall Relative Error: {rel_error:.1e}", 
               ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    plt.suptitle(f"Fisher Information Matrix for Categorical Distribution with probability parameters {round(probs,3)} \n Number of samples = {n_samples}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17)  # Make room for the annotation
    plt.savefig(f"categorical_fisher_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", dpi=500)
    plt.show()

if __name__ == "__main__":
    verify_fisher_categorical()