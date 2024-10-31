import numpy as np

def svd_calc(A):
    
    # Create the square matrix for a left a right side 
    # Right Singular Vector
    A_T_A = A.T @ A
    # Left Singular Vector
    A_A_T = A @ A.T
    
    # Find the eigens and eigenvectors
    # Returns the V
    eigens_ATA, V = np.linalg.eig(A_T_A)
    # Returns the U
    eigens_AAT, U = np.linalg.eig(A_A_T)
    
    # Sort eigenvalues and corresponding eigenvectors
    idx = np.argsort(eigens_ATA)[::-1]  # Sort in descending order
    Eigenvalues_list = eigens_ATA[idx].tolist()  # Sorted eigenvalues
    V = V[:, idx]  # Sorted eigenvectors

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
    S_inv = np.zeros_like(S)
    for i in range(len(singular_values)):
        if singular_values[i] != 0:
            S_inv[i, i] = 1.0 / singular_values[i]

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
        for i in range(springs-1):
            A[i,i] = 1
            A[i+1,i] = -1

    
    # 1 fixed end and free ends systems
    elif (fixed_ends == 1) | (fixed_ends == 0):
        for i in range(springs-1):
            A[i,i] = 1
            if (i+1) < num_masses:
                A[i,i+1] = -1

    
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
    
    # Calculate elongations and internal forces (stresses)
    e = A @ u  # Elongations in each spring
    w = c @ e  # Internal forces (stresses) in each spring
        
    return k, condition_number, V, S, f, u, eigen_values, e, w
        
        
        
        
        


# System parameters
masses = np.array([1.0, 2.0, 1.5])          # Masses in kg
spring_constants = np.array([100, 150,150,100])      # Spring constants in N/m
springs = 4
fixed_ends = 1                               # 1 fixed end, 1 free end

# Call the function
K, condition_number, V, S, f, u, eigen_values, e, w = spring_mass(masses, springs, fixed_ends, spring_constants)

# Display results
print("Stiffness matrix K:\n", K)
print("Condition number of K:", condition_number)
print("Singular values (S):\n", S)
print("Force vector (f):\n", f)
print("Displacement vector (u):\n", u)
print("Elongation vector (e):\n", e)
print("Internal force (stress) vector (w):\n", w)
print("Eigenvalues of K:\n", eigen_values)
