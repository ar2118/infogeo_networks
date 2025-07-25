# Training hyperparameters
NUMBER_EPOCH = 5000
LEARNING_RATE = 0.006
NUMBER_POINTS_ON_SPIRAL = 500
NUMBER_POINTS_USED_FOR_RICCI = 10
NUMBER_SAMPLES_MNIST = 80

NO_RICCI = False  # Set to True for debugging mode, False for normal execution

# Model configuration
NUM_CLASSES = 2
HIDDEN_SIZES = [2]  # Example sizes for hidden layers
ACT_FUNCTION = "tanh"  # "sigmoid", "tanh", "relu", "elu", "leaky_relu", "swish", "gelu", "softplus", "softsign", "silu"

# Numerical settings
RANK_TOL = 1e-15  # Default tolerance, matches the one used in plot_results
KEY_MANUAL = 423242

# Supported activation functions
SUPPORTED_ACTIVATIONS = [
    "sigmoid", "tanh", "relu", "elu", "leaky_relu", 
    "swish", "gelu", "softplus", "softsign", "silu"
]