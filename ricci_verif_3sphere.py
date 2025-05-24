import jax
import jax.numpy as jnp
from jax import jacrev
import matplotlib.pyplot as plt
from datetime import datetime
import scienceplots 
plt.style.use(['science', 'no-latex'])
jax.config.update("jax_enable_x64", True)

def sphere3_metric(x, a=1.0):
    r, theta, _ = x
    return jnp.array([
        [a**2, 0.0, 0.0],
        [0.0, a**2 * jnp.sin(r)**2, 0.0],
        [0.0, 0.0, a**2 * jnp.sin(r)**2 * jnp.sin(theta)**2]
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

def compute_ricci_scalar_sphere3_numerical(a=1.0):
    x0 = jnp.array([jnp.pi / 3, jnp.pi / 4, jnp.pi / 5])  # r, θ, φ
    g = sphere3_metric(x0, a=a)
    g_inv = jnp.linalg.inv(g)

    dg = jacrev(lambda x: sphere3_metric(x, a=a))(x0)
    Gamma = christoffel_symbols_NEW(dg, g_inv)

    def christoffel_fn(x):
        g_x = sphere3_metric(x, a=a)
        g_inv_x = jnp.linalg.inv(g_x)
        dg_x = jacrev(lambda x_: sphere3_metric(x_, a))(x)
        return christoffel_symbols_NEW(dg_x, g_inv_x)

    dGamma = jacrev(christoffel_fn)(x0)
    Ricci = ricci_tensor(Gamma, dGamma)
    ricci_scalar = jnp.einsum('ij,ji->', g_inv, Ricci)
    return ricci_scalar

def compute_ricci_scalar_analytical_sphere3(a):
    return 6 / (a**2)

def plot_ricci_scalars_torus():
    r_values = jnp.linspace(0.2, 1.5, 20)  # Reduced number of points for faster computation
    
    # Print a few test values to debug
    a = 1.0
    R_numeric = compute_ricci_scalar_sphere3_numerical(a)
    R_analytic = compute_ricci_scalar_analytical_sphere3(a)
    print(f"For a={a}, Numerical R={R_numeric}, Analytical R={R_analytic}, Ratio={R_numeric / R_analytic}")

    R_numeric = jnp.array([compute_ricci_scalar_sphere3_numerical(r) for r in r_values])
    R_analytic = jnp.array([compute_ricci_scalar_analytical_sphere3(r) for r in r_values])

    plt.figure(figsize=(9, 6))
    plt.plot(r_values, R_numeric, 'o-', label="Numerical Ricci Scalar", lw=2)
    plt.plot(r_values, R_analytic, 's--', color='magenta', label="Analytical Ricci Scalar", lw=2, markersize=2)
    plt.xlabel("Sphere radius $r$", fontsize=14)
    plt.ylabel("Ricci scalar $R$", fontsize=14)
    plt.title(f"Ricci scalar of a 3-sphere vs radius", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"sphere_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", dpi=500)
    plt.tight_layout()
    
    # Plot the ratio to better understand the discrepancy
    plt.figure(figsize=(10, 6))
    ratio = R_numeric / R_analytic
    plt.plot(r_values, ratio, 'o-', lw=2)
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.xlabel("3-Sphere tube radius $r$")
    plt.ylabel("Ratio (Numerical/Analytical)")
    plt.title("Ratio of Numerical to Analytical Ricci Scalar")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()

# Run the calculation
plot_ricci_scalars_torus()