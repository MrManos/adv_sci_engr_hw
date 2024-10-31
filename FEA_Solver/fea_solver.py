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
    # ||A||*||A^-1|| can be written as max/min of the matrix values as shown in L8
    condition_number = max(singular_values) / min(singular_values)
    # S_inv
    S_inv = np.linalg.inv(S)

    # A^{-1} = V * S^{-1} * U^T
    A_inv = np.dot(np.dot(V, S_inv), U.T)
    
    return np.array(U), S, V, condition_number, A_inv
    


def spring_mass(masses: np.array, springs: int, fixed_ends: int, spring_constants: np.array):
    # masses is a vector of the mass of the balls
    num_masses = len(masses)
    
    # The user inputs a vector of the spring constants
    c = np.diag(spring_constants)

    # Creates the difference matrix based on how many fixed ends there are
    A = np.zeros(( springs, num_masses))

    # 2 fixed ends
    if (fixed_ends == 2):
        for i in range(springs):
            A[i,i] = 1
            A(i+1,i) = -1
            return A
    
    # 1 fixed end and free ends systems
    if (fixed_ends == 1) | (fixed_ends == 0):
        for i in range(springs):
            A[i,i] = 1
            if (i+1) < num_masses:
                A[i,i+1] = -1
            return A
        
    # Creates the element vector
    e = np.array([num_masses,1])
    
    # Creates the displacement matrix
    u = np.array([springs,1])
    
    # Changed this 
    if (type(c) != np.array):
        print(f"Your spring constant needs to be a diagonal matrix")
    
    # Creates the internal force equation
    w = c @ e
    
    # Creates the stiffness matrix k
    k = A.T @ c @ A
    
    _,S,V,condition_number,K_inv = svd_calc(k)
    
    if (K_inv == None):
        raise ValueError("The system matrix K could not be inversed")
    # Eigen values are **2 the singular values
    eigen_values = S **2
    # Creates the force matrix 
    f = masses * 9.81
    
    # Solves the equation u = k^-1*f and gets the displacement
    u = K_inv @ f
        
    return k, condition_number, V, S, f, u, eigen_values
        
        
        
        
        







# Example 
A = np.array([[1.0, 2.0, 3.0], [0.0,1.0, 4.0], [5.0,6.0,0.0]])
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
    