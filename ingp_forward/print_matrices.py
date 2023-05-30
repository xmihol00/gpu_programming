import numpy as np

weights = np.load("tests/net.0.weight.npy")
np.savetxt("weights_l0.csv", weights, delimiter=",", fmt="% 0.7f")
print(f"weights_l0.csv: {weights.shape}")

weights = np.load("tests/net.1.weight.npy")
np.savetxt("weights_l1.csv", weights, delimiter=",", fmt="% 0.7f")
print(f"weights_l1.csv: {weights.shape}")

weights = np.load("tests/net.2.weight.npy")
np.savetxt("weights_l2.csv", weights, delimiter=",", fmt="% 0.7f")
print(f"weights_l2.csv: {weights.shape}")

weights = np.load("tests/5x5/layeroutputs.0.npy")
np.savetxt("output_l0.csv", weights, delimiter=",", fmt="% 0.7f")
print(f"output_l0.csv: {weights.shape}")

weights = np.load("tests/5x5/layeroutputs.1.npy")
np.savetxt("output_l1.csv", weights, delimiter=",", fmt="% 0.7f")
print(f"output_l1.csv: {weights.shape}")

weights = np.load("tests/5x5/layeroutputs.2.npy")
np.savetxt("output_l2.csv", weights, delimiter=",", fmt="% 0.7f")
print(f"output_l2.csv: {weights.shape}")

weights = np.load("tests/5x5/enc_inputs.npy")
np.savetxt("inputs.csv", weights, delimiter=",", fmt="% 0.7f")
print(f"inputs.csv: {weights.shape}")
