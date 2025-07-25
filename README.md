Code to plot the Ricci, Kretschmann and Weyl scalars of a neural network via the Fisher metric, for a synthetic XOR spiral (one hot encoded) and a downscaled MNIST dataset. 

HOW TO USE:
- go to config.py and only edit things there
- if you want to debug use no_ricci = True, this will ignore any ricci computations and just train the network normally (for debugging)
- the 'verif' files contain code to verify the Ricci scalar code and the Fisher matrix code on known analytical results
