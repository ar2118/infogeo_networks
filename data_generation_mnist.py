import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds

from config import NUMBER_SAMPLES_MNIST as NUM_SAMPLES_MNIST

def load_mnist_8_9(split='train', num_samples=NUM_SAMPLES_MNIST, seed=0):
    """
    Load MNIST dataset, filtering for digits 8 and 9, and downsample to 4x4.
    
    Args:
        split (str): Dataset split ('train' or 'test')
        num_samples (int): Number of samples per class to load
        seed (int): Random seed for shuffling
        
    Returns:
        tuple: (X, y) where X is the input data and y is one-hot encoded labels
    """
    ds, _ = tfds.load("mnist", split=split, as_supervised=True, with_info=True)
    print(f"Dataset loaded: {split} split")
    ds = ds.shuffle(10_000, seed=seed)

    X_list, y_list = [], []
    counts = {8: 0, 9: 0}  # To track how many of each digit we've collected
    
    for img, label in tfds.as_numpy(ds):
        # Only process if the label is 8 or 9
        if label not in [8, 9]:
            continue
            
        # Check if we've collected enough samples of this digit
        if counts[label] >= num_samples:
            # If we have enough of both classes, break
            if all(count >= num_samples for count in counts.values()):
                break
            continue
            
        # Process the image
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        img = img.reshape(28, 28)  # Original shape
        
        # Downsample to 4×4 using bilinear interpolation
        img_small = jax.image.resize(img, shape=(4, 4), method="bilinear")
        
        # Create one-hot encoding (2 classes: 8->0, 9->1)
        y_onehot = np.zeros(2, dtype=np.float32)
        y_onehot[label - 8] = 1.0  # Map 8->0, 9->1
        
        X_list.append(img_small.flatten())  # flatten to (16,)
        y_list.append(y_onehot)
        
        counts[label] += 1

    X = jnp.array(X_list)
    y = jnp.array(y_list)
    
    print(f"Loaded {len(X)} samples with shape {X.shape}")
    print(f"Class distribution: {counts}")
    
    return X, y

def create_test_data():
    """Create a small test dataset for debugging."""
    # Generate some simple test data
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (20, 16))  # 20 samples, 16 features
    
    # Create simple binary labels
    y = jnp.zeros((20, 2))
    y = y.at[:10, 0].set(1.0)  # First 10 samples are class 0
    y = y.at[10:, 1].set(1.0)  # Last 10 samples are class 1
    
    return X, y

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