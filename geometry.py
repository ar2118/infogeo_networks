"""
Geometric analysis utilities for computing Fisher information and curvature tensors.
"""
import jax
import jax.numpy as jnp
from jax import jacrev, vmap, grad
from jax.flatten_util import ravel_pytree
import numpy as np
from model import forward
from config import NUM_CLASSES, RANK_TOL, NUMBER_POINTS_USED_FOR_RICCI
rank_tol = RANK_TOL  # Tolerance for rank calculation
num_classes= NUM_CLASSES  # Number of classes in the model

def fisher(params, x):
    """
    Compute the Fisher Information Matrix.
    
    Args:
        params: Model parameters
        x: Input data
        
    Returns:
        Fisher information matrix
    """
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


def calculate_rank(matrix, tol=rank_tol):
    """Calculate the rank of a matrix based on SVD and a threshold."""
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    rank = jnp.sum(singular_values > tol)
    return rank


def reduce_matrix_dimensions_by_threshold(matrix, threshold=rank_tol):
    """
    Reduce matrix dimensions by eigenvalue threshold.
    
    Args:
        matrix: Input matrix to reduce
        threshold: Eigenvalue threshold
        
    Returns:
        tuple: (reduced_matrix, V, V_T) where V is projection matrix
    """
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


@jax.jit
def christoffel_symbols_NEW(dg, g_inv):
    """Compute Christoffel symbols from metric derivatives."""
    partial_derivs = jnp.einsum('mns -> smn', dg)
    sum_partial_derivs = partial_derivs + jnp.einsum('nrm -> mnr', partial_derivs) - jnp.einsum('rmn -> mnr', partial_derivs)
    christ = 0.5 * jnp.einsum('sr, mnr -> smn', g_inv, sum_partial_derivs)
    return christ

@jax.jit
def riemann_curvature(Gamma, dGamma):
    """Compute the Riemann curvature tensor."""
    dGamma = jnp.einsum('rmns -> srmn', dGamma) # Rearranging indices because when we differentiate we get the extra index at the END
    term1 = jnp.einsum('mrns -> rsmn', dGamma)
    term2 = jnp.einsum('nrms -> rsmn', dGamma)
    term3 = jnp.einsum('rml, lns -> rsmn', Gamma, Gamma)
    term4 = jnp.einsum('rnl, lms -> rsmn', Gamma, Gamma)
    return term1 - term2 + term3 - term4

@jax.jit
def ricci_tensor(Gamma, dGamma):
    """Compute the Ricci tensor."""
    riemann = riemann_curvature(Gamma, dGamma)
    return jnp.einsum('rsru -> su', riemann)


@jax.jit
def ricci_tensor_fast(Gamma, dGamma):
    """Fast computation of Ricci tensor."""
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


def weyl_tensor(g, riemann, ricci_tensor_val, ricci_scalar):
    """Compute the Weyl tensor."""
    n = g.shape[0]

    # First term: Riemann itself
    C = riemann

    # Second term (with Ricci tensor and metric)
    term2 = (1 / (n - 2)) * (
        jnp.einsum('im,kl->iklm', ricci_tensor_val, g)
    - jnp.einsum('il,km->iklm', ricci_tensor_val, g)
    + jnp.einsum('kl,im->iklm', ricci_tensor_val, g)
    - jnp.einsum('km,il->iklm', ricci_tensor_val, g)
    )

    # Third term (with Ricci scalar and metric)
    term3 = (ricci_scalar / ((n - 1) * (n - 2))) * (
        jnp.einsum('il,km->iklm', g, g)
    - jnp.einsum('im,kl->iklm', g, g)
    )

    return C + term2 + term3


def weyl_scalar(g_inv, C):
    """Compute the Weyl scalar."""
    # Raise all indices on C_{ijkl} to get C^{ijkl}
    C_up = jnp.einsum('im,jn,kp,lq,mnpq->ijkl',
                    g_inv, g_inv, g_inv, g_inv, C)

    # Contract C_{ijkl} with C^{ijkl}
    C2 = jnp.einsum('ijkl,ijkl->', C, C_up)
    return C2


def compute_ricci_tensor_from_fisher(params, x, constant_V=None, constant_V_T=None):
    """
    Compute geometric quantities from Fisher information matrix.
    
    Args:
        params: Model parameters
        x: Input data
        constant_V: Constant projection matrix (optional)
        constant_V_T: Transpose of constant projection matrix (optional)
        
    Returns:
        tuple: (ricci_scalar, kretschmann_scalar, weyl_scalar)
    """
    flat_params, unravel_fn = ravel_pytree(params)

    # ---- Metric: Fisher matrix ---- 
    # This extra internal function is necessary to get the Fisher via flattened params
    @jax.jit
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
    @jax.jit
    def reparam_model(reduced_params):
        full_params_flat = projection_matrix @ reduced_params  # shape (N,)
        return unravel_fn(full_params_flat)

    def log_prob_reduced(reduced_params, x_i, c):
        params_dict = reparam_model(reduced_params)
        probs = forward(params_dict, x_i.reshape(1, -1)).squeeze()
        return jnp.log(probs[c])

    def per_sample_fisher_reduced(reduced_params, x_i):
        p_i = forward(reparam_model(reduced_params), x_i.reshape(1, -1)).squeeze()
        contrib = jnp.zeros((reduced_params.shape[0], reduced_params.shape[0]))
        for c in range(num_classes):
            J_c = grad(log_prob_reduced)(reduced_params, x_i, c)
            contrib += p_i[c] * jnp.outer(J_c, J_c)
        return contrib

    @jax.jit
    def fisher_reduced(reduced_params, x_data):
        return jnp.mean(vmap(per_sample_fisher_reduced, in_axes=(None, 0))(reduced_params, x_data), axis=0)

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

    @jax.jit
    def christoffel_fn(reduced_p, x):
        g_x = fisher_reduced(reduced_p, x)
        g_inv_x = jnp.linalg.inv(g_x)
        dg_x = jacrev(fisher_reduced)(reduced_p, x)    # (d, d, d): ∂g_ij / ∂θ_k
        return christoffel_symbols_NEW(dg_x, g_inv_x)

    dGamma = jacrev(christoffel_fn)(reduced_params0, x)
    ricci = ricci_tensor(Gamma, dGamma)
    ricci = (ricci + ricci.T) / 2  # Ensure symmetry
    ricci_scalar = jnp.einsum('ij,ji->', g_inv, ricci)
    riemann = riemann_curvature(Gamma, dGamma)
    kretschmann_scalar_value = kretschmann_scalar(g_inv, riemann)
    C = weyl_tensor(g, riemann, ricci, ricci_scalar)
    weyl_scalar_value = weyl_scalar(g_inv, C)

    return ricci_scalar, kretschmann_scalar_value, weyl_scalar_value