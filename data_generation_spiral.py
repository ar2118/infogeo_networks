"""
Data generation module for XOR spiral task.
"""
import numpy as np
import jax.numpy as jnp
from config import NUMBER_POINTS_ON_SPIRAL

def generate_spiral_data(n_total=NUMBER_POINTS_ON_SPIRAL, noise=0.05, seed=0):
    """
    Generate two intertwined spirals with XOR pattern.
    
    Args:
        n_total (int): Total number of points to generate
        noise (float): Amount of noise to add to the spirals
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (X, y, centers, colors) where:
            - X: Input features (n_total, 2)
            - y: One-hot encoded labels (n_total, 2)  
            - centers: Empty list (for compatibility)
            - colors: List of colors for plotting
    """
    np.random.seed(seed)
    X_list, y_list, colors = [], [], []
    n_per_class = int(n_total * 0.5)
    theta = np.linspace(0, 1.5 * np.pi, n_per_class)

    # Spiral 1 (class 0)
    r = theta
    x1 = r * np.cos(theta) + np.random.normal(0, noise, size=theta.shape)
    y1 = r * np.sin(theta) + np.random.normal(0, noise, size=theta.shape)
    data1 = np.stack([x1, y1], axis=1)
    labels1 = np.tile([1, 0], (n_per_class, 1))
    color1 = ['red'] * n_per_class

    # Spiral 2 (class 1), shifted by π
    x2 = r * np.cos(theta + np.pi) + np.random.normal(0, noise, size=theta.shape)
    y2 = r * np.sin(theta + np.pi) + np.random.normal(0, noise, size=theta.shape)
    data2 = np.stack([x2, y2], axis=1)
    labels2 = np.tile([0, 1], (n_per_class, 1))
    color2 = ['blue'] * n_per_class

    # Combine
    X = jnp.array(np.vstack([data1, data2]), dtype=jnp.float64)
    y = jnp.array(np.vstack([labels1, labels2]), dtype=jnp.float64)
    colors = color1 + color2

    print(f"Generated two intertwined spirals with {X.shape[0]} points.")
    return X, y, [], colors


def subsample_x(x, max_points):
    """
    Subsample data points for Ricci computation.
    
    Args:
        x: Input data
        max_points: Maximum number of points to keep
        
    Returns:
        Subsampled data
    """
    return x[:max_points]