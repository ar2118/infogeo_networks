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

def split_params(params):
    params_reduced = params.copy()
    params_frozen = {}

    W_out = params["W_out"]
    b_out = params["b_out"]

    # Split W_out: keep first column in reduced, second in frozen
    params_reduced["W_out"] = W_out[:, :1]  # shape (hidden_dim, 1)
    params_frozen["W_out"] = W_out[:, 1:]   # shape (hidden_dim, 1)

    # Same for b_out
    params_reduced["b_out"] = b_out[:1]
    params_frozen["b_out"] = b_out[1:]

    return params_reduced, params_frozen

def merge_params(params_reduced, params_frozen):
    merged = params_reduced.copy()

    # Concatenate W_out columns
    merged["W_out"] = jnp.concatenate(
        [params_reduced["W_out"], params_frozen["W_out"]],
        axis=1
    )
    # Concatenate b_out
    merged["b_out"] = jnp.concatenate(
        [params_reduced["b_out"], params_frozen["b_out"]],
        axis=0
    )

    return merged

def fisher_reduced(params, x):
    # 1. Split params
    params_reduced, params_frozen = split_params(params)

    # 2. Flatten reduced params
    flat_reduced, unravel_reduced = ravel_pytree(params_reduced)
    num_reduced_params = flat_reduced.shape[0]

    # 3. Define log_probs_fn with only reduced params variable
    def log_probs_fn(flat_reduced, x_i, c):
        # Reconstruct reduced param pytree
        params_reduced_ = unravel_reduced(flat_reduced)
        # Merge back with frozen params
        merged_params = merge_params(params_reduced_, params_frozen)
        # Compute log prob
        probs = forward(merged_params, x_i.reshape(1, -1)).squeeze()
        return jnp.log(probs[c])

    # 4. Per-sample Fisher
    def per_sample_fisher(flat_reduced, x_i):
        probs_i = forward(merge_params(unravel_reduced(flat_reduced), params_frozen), x_i.reshape(1, -1)).squeeze()
        contrib = jnp.zeros((num_reduced_params, num_reduced_params))
        for c in range(num_classes):
            J_c = jacrev(log_probs_fn)(flat_reduced, x_i, c)
            contrib += probs_i[c] * jnp.outer(J_c, J_c)
        return contrib

    # 5. Vectorize over all samples
    fisher_samples = vmap(per_sample_fisher, in_axes=(None, 0))(flat_reduced, x)
    fisher_matrix = jnp.mean(fisher_samples, axis=0)

    return fisher_matrix


def calculate_rank(matrix, tol=rank_tol):
    """Calculate the rank of a matrix based on SVD and a threshold."""
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    rank = jnp.sum(singular_values > tol)
    return rank

@jax.jit
def christoffel_symbols_NEW(dg, g_inv):
    partial_derivs = jnp.einsum('mns -> smn', dg)
    sum_partial_derivs = partial_derivs + jnp.einsum('nrm -> mnr', partial_derivs) - jnp.einsum('rmn -> mnr', partial_derivs)
    christ = 0.5 * jnp.einsum('sr, mnr -> smn', g_inv, sum_partial_derivs)
    return christ

def riemann_curvature(Gamma, dGamma):
    dGamma = jnp.einsum('rmns -> srmn', dGamma) # Rearranging indices because when we differentiate we get the extra index at the END
    term1 = jnp.einsum('mrns -> rsmn', dGamma)
    term2 = jnp.einsum('nrms -> rsmn', dGamma)
    term3 = jnp.einsum('rml, lns -> rsmn', Gamma, Gamma)
    term4 = jnp.einsum('rnl, lms -> rsmn', Gamma, Gamma)
    return term1 - term2 + term3 - term4

def ricci_tensor(Gamma, dGamma):
    riemann = riemann_curvature(Gamma, dGamma)
    return jnp.einsum('rsru -> su', riemann)

def kretschmann_scalar(g_inv, R):
    R_up = jnp.einsum('im,jn,kp,lq,mnpq->ijkl',
                    g_inv, g_inv, g_inv, g_inv, R)
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


def compute_ricci_tensor_from_fisher(params, x):

    params_reduced, params_frozen = split_params(params)
    flat_reduced, unravel_reduced = ravel_pytree(params_reduced)

    def fisher_function_for_differentiation(flat_reduced, params_frozen, unravel_reduced, x):
        num_reduced_params = flat_reduced.shape[0]

        def log_probs_fn(flat_reduced, x_i, c):
            params_reduced_ = unravel_reduced(flat_reduced)
            merged_params = merge_params(params_reduced_, params_frozen)
            probs = forward(merged_params, x_i.reshape(1, -1)).squeeze()
            return jnp.log(probs[c])

        def per_sample_fisher(flat_reduced, x_i):
            probs_i = forward(merge_params(unravel_reduced(flat_reduced), params_frozen), x_i.reshape(1, -1)).squeeze()
            contrib = jnp.zeros((num_reduced_params, num_reduced_params))
            for c in range(num_classes):
                J_c = jacrev(log_probs_fn)(flat_reduced, x_i, c)
                contrib += probs_i[c] * jnp.outer(J_c, J_c)
            return contrib

        fisher_samples = vmap(per_sample_fisher, in_axes=(None, 0))(flat_reduced, x)
        fisher_matrix = jnp.mean(fisher_samples, axis=0)

        return fisher_matrix
    
    # Get my reduced parameters (without redundancies)
    params_reduced, params_frozen = split_params(params)
    flat_reduced, unravel_reduced = ravel_pytree(params_reduced)

    # Calculate REDUCED Fisher matrix, its invers and its derivative with respect to the reduced parameters
    g = fisher_function_for_differentiation(flat_reduced, params_frozen, unravel_reduced, x)
    g = (g + g.T) / 2 # Ensure symmetry in case of small numerical precision errors, this isn't even that needed and I have checked (they blow up a bit sometimes though)
    g_inv = jnp.linalg.inv(g)
    g_inv = (g_inv + g_inv.T) / 2
    dg = jacrev(fisher_function_for_differentiation)(flat_reduced, params_frozen, unravel_reduced, x)  # shape (d, d, d)

    # Make a function for christoffel symbols so that I can use it in the derivatives using jacrev
    def Gamma_from_dg_ginv(dg, g_inv):
        return christoffel_symbols_NEW(dg, g_inv)  # shape (d, d, d)
    
    # Derivatives: ∂Gamma / ∂dg and ∂Gamma / ∂g_inv (linear maps) - doing this instead of jacrev to speed up compute (i.e. using chain rule on Gamma)
    Gamma_wrt_dg = jax.jacrev(Gamma_from_dg_ginv, argnums=0)(dg, g_inv)      # shape (d,d,d, d,d, d)
    Gamma_wrt_ginv = jax.jacrev(Gamma_from_dg_ginv, argnums=1)(dg, g_inv)    # shape (d,d,d, d,d)

    # Derivatives of inputs w.r.t. theta, this is the chain rule part that i multiply with the two above
    dg_wrt_theta = jacrev(jacrev(fisher_function_for_differentiation))(flat_reduced, params_frozen, unravel_reduced, x)                  # shape (d,d,d)
    g_inv_wrt_theta = jacrev(lambda t: jnp.linalg.inv(fisher_function_for_differentiation(t, params_frozen, unravel_reduced, x)))(flat_reduced)  # shape (d,d,d)

    # Compute and add up both contributions from the chain rule
    dGamma_from_dg = jnp.einsum("abcijk,ijkl -> abcl", Gamma_wrt_dg, dg_wrt_theta)
    dGamma_from_ginv = jnp.einsum("abcij,ijl -> abcl", Gamma_wrt_ginv, g_inv_wrt_theta)
    dGamma_total = dGamma_from_dg + dGamma_from_ginv  # shape (d,d,d,theta_dim)

    Gamma = christoffel_symbols_NEW(dg, g_inv)  # shape (d,d,d)

    ricci = ricci_tensor(Gamma, dGamma_total)
    ricci = (ricci + ricci.T) / 2  # Ensure symmetry
    ricci_scalar_value = jnp.einsum('ij,ji->', g_inv, ricci)

    # Calculate other scalars
    kretschmann_scalar_value = kretschmann_scalar(g_inv, riemann_curvature(Gamma, dGamma_total))
    weyl_tensor_result = weyl_tensor(g, riemann_curvature(Gamma, dGamma_total), ricci, ricci_scalar_value)
    weyl_scalar_value = weyl_scalar(g_inv, weyl_tensor_result)

    return ricci_scalar_value, kretschmann_scalar_value, weyl_scalar_value, g
