import numpy as np

def svd_calc(A):
    
    # Create the square matrix for a left a right side 
    # Right Singular Vector
    A_T_A = np.dot(A.T, A)
    # Left Singular Vector
    A_A_T = np.dot(A,A.T)
    
    # Find the eigens and eigenvectors
    # Returns the V
    eigens_ATA, V = np.linalg.eig(A_T_A)
    # Returns the U
    eigens_AAT, U = np.linalg.eig(A_A_T)
    
    # Create a list of eigenvalues/vectors
    Eigenvalues_list = eigens_ATA.tolist()
    V_list = V.tolist()

    # set up bubble sort
    n = len(Eigenvalues_list)
    for i in range(n):
        # Find largest eigenvalue index
        max_idx = i
        for j in range(i + 1, n):
            if Eigenvalues_list[j] > Eigenvalues_list[max_idx]:
                max_idx = j
    
        # Swap the eigenvalues
        Eigenvalues_list[i], Eigenvalues_list[max_idx] = Eigenvalues_list[max_idx], Eigenvalues_list[i]
    
        # Swap the corresponding eigenvectors
        V[i], V[max_idx] = V[max_idx], V[i]
    
    # Convert V back to an array and transpose
    V = np.array(V).T

    # Remove any tiny negative values due to numerical precision
    singular_values = [np.sqrt(abs(eigval)) for eigval in Eigenvalues_list]

    # Form the singular value matrix S (m x n with singular values on the diagonal)
    S = np.zeros_like(A, dtype=float)
    for i in range(min(len(singular_values), S.shape[1])):
        S[i, i] = singular_values[i]

    # Check if any singular values are 0, this means its non-invertible
    if any(s == 0 for s in singular_values):
        raise ValueError("Matrix is non-invertible due to zero singular values.")

    # Condition number is the largest/smallest singular value
    condition_number = max(singular_values) / min(singular_values)

    # S_inv
    S_inv = np.linalg.inv(S)

    # A^{-1} = V * S^{-1} * U^T
    A_inv = np.dot(np.dot(V, S_inv), U.T)
    
    return np.array(U), S, V, condition_number, A_inv
    
    
# Example 
A = np.array([[1, 2], [3, 4]], dtype=float)
try:
    U, S, Vt, cond_num, A_inv = svd_calc(A)
    print("U matrix:\n", U)
    print("S matrix (singular values):\n", S)
    print("V^T matrix:\n", Vt)
    print("Condition number:", cond_num)
    print("Inverse matrix:\n", A_inv)
except ValueError as e:
    print(e)
    
    
print(f"This seperates my svd and the regular svd")
[U,S,Vh] = np.linalg.svd(A)
print("U matrix:\n", U)
print("S matrix (singular values):\n" , S)
print("V^T matrix:\n", Vh)
    