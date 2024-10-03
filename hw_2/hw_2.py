import numpy as np

# Define the matrix A
# A = np.array([[-3, 1],
#               [6, -2],
#               [6, -2]])

A = np.array([[3,2,2],[2,3,-2]])
# Perform Singular Value Decomposition
U, Sigma, Vt = np.linalg.svd(A)

# Output the results
print("Matrix A:")
print(A)

print("\nLeft singular vectors (U):")
print(U)

print("\nSingular values (Sigma):")
print(Sigma)

print("\nRight singular vectors (V^T):")
print(Vt)

# To show right singular vectors (V) instead of V^T
print("\nRight singular vectors (V):")
V = Vt.T  # Transpose Vt to get V
print(V)

# Reconstruct A using U, Sigma, and Vt
# Create a full Sigma matrix with the correct shape
Sigma_full = np.zeros((A.shape[0], A.shape[1]))  # Create a full Sigma matrix
np.fill_diagonal(Sigma_full, Sigma)  # Fill the diagonal with singular values

A_reconstructed = np.dot(U, np.dot(Sigma_full, Vt))  # U * Sigma * V^T

print("\nReconstructed Matrix A (using U, Sigma, V^T):")
print(A_reconstructed)
