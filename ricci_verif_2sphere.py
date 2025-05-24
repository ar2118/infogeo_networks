import jax
import jax.numpy as jnp
from jax import jacrev
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

def sphere_metric(x, r=1.0):
    theta = x[0]
    return jnp.array([
        [r**2, 0.0],
        [0.0, r**2 * jnp.sin(theta)**2]
    ])

def christoffel_symbols_NEW(dg, g_inv):
    partial_derivs = jnp.einsum('mns -> smn', dg)
    sum_partial_derivs = partial_derivs + jnp.einsum('nrm -> mnr', partial_derivs) - jnp.einsum('rmn -> mnr', partial_derivs)
    christ = 0.5 * jnp.einsum('sr, mnr -> smn', g_inv, sum_partial_derivs)
    return christ

def ricci_tensor(Gamma, dGamma):
    dGamma = jnp.einsum('rmns -> srmn', dGamma) # Rearranging indices because when we differentiate we get the extra index at the END
    term1 = jnp.einsum('mrns -> ns', dGamma)               
    term2 = jnp.einsum('nrms -> ns', dGamma)                
    term3 = jnp.einsum('rml, lns -> ns', Gamma, Gamma)              
    term4 = jnp.einsum('rnl, lms -> ns', Gamma, Gamma)      
    return term1 - term2 + term3 - term4

def compute_ricci_scalar_sphere_numerical(r=1):
    x0 = jnp.array([jnp.pi / 4, jnp.pi / 4])  # θ, φ
    g = sphere_metric(x0, r=r)
    g_inv = jnp.linalg.inv(g)

    # Compute metric derivatives using JAX's automatic differentiation
    dg = jacrev(lambda x: sphere_metric(x, r=r))(x0)
    Gamma = christoffel_symbols_NEW(dg, g_inv)

    def christoffel_fn(x):
        g_x = sphere_metric(x, r=r)
        g_inv_x = jnp.linalg.inv(g_x)
        dg_x = jacrev(lambda x_: sphere_metric(x_, r))(x)
        return christoffel_symbols_NEW(dg_x, g_inv_x)

    # Compute derivatives of Christoffel symbols
    dGamma = jacrev(christoffel_fn)(x0)
    Ricci = ricci_tensor(Gamma, dGamma)
    ricci_scalar = jnp.einsum('ij,ji->', g_inv, Ricci)  # Changed to g_inv_ij * R_ji
    return ricci_scalar

def compute_ricci_scalar_analytical_sphere(r):
    # The analytical formula for the Ricci scalar of a torus
    return 2 / (r**2)

def plot_ricci_scalars_torus():
    r_values = jnp.linspace(0.2, 1.5, 20)  # Reduced number of points for faster computation
    
    # Print a few test values to debug
    test_r = 1.0
    numeric = compute_ricci_scalar_sphere_numerical(r=test_r)
    analytic = compute_ricci_scalar_analytical_sphere(r=test_r)
    print(f"For r={test_r}, Numerical R={numeric}, Analytical R={analytic}, Ratio={numeric/analytic}")

    R_numeric = jnp.array([compute_ricci_scalar_sphere_numerical(r) for r in r_values])
    R_analytic = jnp.array([compute_ricci_scalar_analytical_sphere(r) for r in r_values])

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, R_numeric, 'o-', label="Numerical Ricci Scalar", lw=2)
    plt.plot(r_values, R_analytic, 's--', label="Analytical Ricci Scalar", lw=2)
    plt.xlabel("Torus tube radius $r$")
    plt.ylabel("Ricci scalar $R$")
    plt.title(f"Ricci Scalar of a sphere at θ = π/3 vs radius")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Plot the ratio to better understand the discrepancy
    plt.figure(figsize=(10, 6))
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