import jax
import jax.numpy as jnp
from jax import jacrev
import matplotlib.pyplot as plt
from datetime import datetime
import scienceplots 
plt.style.use(['science', 'no-latex'])
jax.config.update("jax_enable_x64", True)

# Torus metric in (theta, phi) coordinates
def torus_metric(x, R=2.0, r=1.0):
    theta = x[0]
    return jnp.array([
        [r**2, 0.0],
        [0.0, (R + r * jnp.cos(theta))**2]
    ])

def christoffel_symbols_NEW(dg, g_inv):
    """
    Compute the Christoffel symbols at a given coordinate.
    """
    partial_derivs = jnp.einsum('mns -> smn', dg)
    sum_partial_derivs = partial_derivs + jnp.einsum('nrm -> mnr', partial_derivs) - jnp.einsum('rmn -> mnr', partial_derivs)
    christ = 0.5 * jnp.einsum('sr, mnr -> smn', g_inv, sum_partial_derivs)
    return christ

def ricci_tensor(Gamma, dGamma):

    dGamma = jnp.einsum('rmns -> srmn', dGamma)             # Rearranging indices because when we differentiate we get the extra index at the END
    term1 = jnp.einsum('mrns -> ns', dGamma)               
    term2 = jnp.einsum('nrms -> ns', dGamma)                
    term3 = jnp.einsum('rml, lns -> ns', Gamma, Gamma)              
    term4 = jnp.einsum('rnl, lms -> ns', Gamma, Gamma)      
    
    # Return Ricci tensor
    return term1 - term2 + term3 - term4

def compute_ricci_scalar_torus_numerical(r_minor, R_major=2.0):
    x0 = jnp.array([jnp.pi / 4, jnp.pi / 4])  # θ, φ
    g = torus_metric(x0, R=R_major, r=r_minor)
    g_inv = jnp.linalg.inv(g)

    # Compute metric derivatives using JAX's automatic differentiation
    dg = jacrev(lambda x: torus_metric(x, R=R_major, r=r_minor))(x0)
    Gamma = christoffel_symbols_NEW(dg, g_inv)

    # Define a function that computes Christoffel symbols at a given point
    def christoffel_fn(x):
        g_x = torus_metric(x, R=R_major, r=r_minor)
        g_inv_x = jnp.linalg.inv(g_x)
        dg_x = jacrev(lambda x_: torus_metric(x_, R=R_major, r=r_minor))(x)
        return christoffel_symbols_NEW(dg_x, g_inv_x)

    # Compute derivatives of Christoffel symbols
    dGamma = jacrev(christoffel_fn)(x0)
    
    # Compute Ricci tensor and scalar
    Ricci = ricci_tensor(Gamma, dGamma)
    ricci_scalar = jnp.einsum('ij,ji->', g_inv, Ricci)  # Changed to g_inv_ij * R_ji
    return ricci_scalar

def compute_ricci_scalar_analytical_torus(theta, R=2.0, r=1.0):
    # The analytical formula for the Ricci scalar of a torus
    return 2 * jnp.cos(theta) / (r * (R + r * jnp.cos(theta)))

def plot_ricci_scalars_torus():
    r_values = jnp.linspace(0.2, 1.5, 20)  # Reduced number of points for faster computation
    R_major = 2.0
    theta = jnp.pi / 4
    
    # Print a few test values to debug
    test_r = 1.0
    numeric = compute_ricci_scalar_torus_numerical(test_r, R_major)
    analytic = compute_ricci_scalar_analytical_torus(theta, R=R_major, r=test_r)
    print(f"For r={test_r}, Numerical R={numeric}, Analytical R={analytic}, Ratio={numeric/analytic}")

    R_numeric = jnp.array([compute_ricci_scalar_torus_numerical(r, R_major) for r in r_values])
    R_analytic = jnp.array([compute_ricci_scalar_analytical_torus(theta, R=R_major, r=r) for r in r_values])

    plt.figure(figsize=(9, 6))
    plt.plot(r_values, R_numeric, 'o-', label="Numerical Ricci Scalar", lw=2)
    plt.plot(r_values, R_analytic, 's--', label="Analytical Ricci Scalar", lw=2, markersize=3)
    plt.xlabel("Torus tube radius $r$", fontsize=14)
    plt.ylabel("Ricci scalar $R$", fontsize=14)
    plt.title(f"Ricci scalar of a Torus vs tube radius", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"torus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", dpi=500)
    plt.tight_layout()
    
    # Plot the ratio to better understand the discrepancy
    plt.figure(figsize=(12, 6))
    ratio = R_numeric / R_analytic
    plt.plot(r_values, ratio, 'o-', lw=2)
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.xlabel("Torus tube radius $r$")
    plt.ylabel("Ratio (Numerical/Analytical)")
    plt.title("Ratio of Numerical to Analytical Ricci Scalar")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()

# Run the calculation
plot_ricci_scalars_torus()