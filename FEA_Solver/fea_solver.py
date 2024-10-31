import numpy as np

def Spring_mass_with_user_inputs():
    # Number of masses
    num_masses = int(input("Enter the number of masses: "))
    
    # Masses in kg
    masses = []
    for i in range(num_masses):
        mass = float(input(f"Enter mass {i + 1} (in kg): "))
        masses.append(mass)
    masses = np.array(masses)
    
    # Number of springs (should be one more than the number of masses)
    springs = num_masses + 1
    
    # Spring constants in N/m
    spring_constants = []
    print(f"Enter the spring constants for each of the {springs} springs:")
    for i in range(springs):
        k = float(input(f"Spring constant {i + 1} (in N/m): "))
        spring_constants.append(k)
    spring_constants = np.array(spring_constants)
    
    # Fixed ends configuration
    print("Enter the fixed ends configuration:")
    print("0 - No fixed ends")
    print("1 - One fixed end")
    print("2 - Two fixed ends")
    fixed_ends = int(input("Enter the number of fixed ends (0,1,2): "))
    
    k, condition_number, V, S_matrix, f, u, eigen_values, e, w = spring_mass(masses=masses, springs=springs, fixed_ends=fixed_ends, spring_constants=spring_constants)
    
    return k, condition_number, V, S_matrix, f, u, eigen_values, e, w



def svd_calc(A):
    """
    Calculates the Singular Value Decomposition of matrix A

    Inputs:
    A (np.array m x n matrix) - matrix set for decomposition

    returns: 
    u (np.array): left singular value
    v (np.array): right singular value
    conditional number (float) 
    A_inv (np.array) - Inverse of matrix A

    """
    # Create the square matrix for a left a right side 
    # Right Singular Vector
    A_T_A = A.T @ A
    # Left Singular Vector
    A_A_T = A @ A.T
    # Find the eigens and eigenvectors
    # Returns the V
    eigens_ATA, V = np.linalg.eig(A_T_A)
    eigens_AAT, U = np.linalg.eig(A_A_T)

    # get the singular vaules
    singular_values = np.sqrt(np.abs(eigens_ATA))

    # Sort eigenvalues and corresponding eigenvectors
    idx = np.argsort(singular_values)[::-1]  # Sort in descending order
    singular_values = singular_values[idx] # Sorted eigenvalues
    V = V[:,idx]  # Sorted eigenvectors

    # Convert V back to SVD form
    V = np.array(V).T

    # Singular value diagonal matrix
    sigma = np.diag(singular_values)

    # Check if any singular values are 0, this means its non-invertible
    if any(s == 0 for s in singular_values):
        raise ValueError("Matrix is non-invertible due to a zero singular value.")

    # Condition number is the largest/smallest singular value
    # ||A||*||A^-1|| can be written as max/min of the matrix values as shown in L8
    condition_number = max(singular_values) / min(singular_values)

    # S_inv
    S_inv = np.zeros_like(sigma)
    for i in range(len(singular_values)):
        if singular_values[i] != 0:
            S_inv[i, i] = 1.0 / singular_values[i]

    # A^{-1} = V * S^{-1} * U^T
    A_inv = V @ S_inv @ U.T
    
    return U, sigma , V, condition_number, A_inv, eigens_ATA
    


def spring_mass(masses: np.array, springs: int, fixed_ends: int, spring_constants: np.array):
    """
    Calculates the stiffness matrix, condition number, displacements, elongations, and internal forces
    for a spring-mass system given the masses, spring constants, and amount of fixed ends
    
    Parameters:
    ----------
    masses (np.array) - A 1D numpy array of the masses attached to each point in the system (in kg).
    
    springs (int) - The number of springs in the system, which should be one more than the number of masses.
    
    fixed_ends (int) - Boundary conditions of the system
    
    spring_constants (np.array) - A 1D numpy array of the spring constants for each spring (in N/m).
    
    Returns:
    -------
    k (np.array) - The stiffness matrix of the system.
    
    condition_number (float) - The condition number of the stiffness matrix, indicating the stability of the system.
    
    V (np.array) - The right singular vectors matrix from the SVD of the stiffness matrix.
    
    S_matrix (np.array) - A diagonal matrix containing the singular values of the stiffness matrix.
    
    f (np.array) - The gravitational force vector on each mass in the system.
    
    u (np.array) - The displacement vector of each mass.
    
    eigen_values (np.array) - The eigenvalues of the stiffness matrix, representing the squared singular values.
    
    e (np.array) - The elongation in each spring, calculated as the difference in displacements between adjacent masses.
    
    w (np.array) -The internal forces or stresses in each spring, based on the elongations.
   
    Raises:
    ------
    ValueError
        If `fixed_ends` is 0 (unsupported configuration), or if there is a dimension mismatch between matrices.
    
    """    
    # masses is a vector of the mass of the balls
    num_masses = len(masses)
    
    # The user inputs a vector of the spring constants
    c = np.diag(spring_constants)

    # Creates the difference matrix based on how many fixed ends there are
    A = np.zeros(( springs, num_masses))

    # 2 fixed ends
    if (fixed_ends == 2):
        for i in range(num_masses):
            A[i,i] = 1
            A[i+1,i] = -1

    
    # 1 fixed end and free ends systems
    elif (fixed_ends == 1):
        for i in range(num_masses):
            A[i,i] = 1
            if (i+1) < num_masses:
                A[i,i+1] = -1

    elif (fixed_ends == 0):
        raise ValueError("This fails because we have we do not have a full rank, this amount of fixed ends has a non zero null space.")
    
    # Ensure that A, c, and the resulting stiffness matrix match in dimensions
    if A.shape[0] != c.shape[0]:
        raise ValueError("Dimension mismatch: A and c should have compatible dimensions.")
    
    
    # Creates the stiffness matrix k
    k = A.T @ c @ A
    
    U, S_matrix, V, condition_number, K_inv, _ = svd_calc(k)
    
        # Creates the element vector
    e = np.array([num_masses,1])
    
    # Creates the displacement matrix
    u = np.array([springs,1])
    
    # Changed this 
    if (type(c) != np.array):
        print(f"Your spring constant needs to be a diagonal matrix")
    
    
     # Check if the condition number indicates an invertible matrix
    if condition_number > 1e12:  # Threshold for singularity
        raise ValueError("The system matrix K is ill-conditioned and close to singular.")

    
    # Eigen values are **2 the singular values
    eigen_values = S_matrix **2
    # Creates the force matrix 
    f = masses * 9.81
    
    # Solves the equation u = k^-1*f and gets the displacement
    u = K_inv @ f
    
    # Calculate elongations and internal forces (stresses)
    e = A @ u  # Elongations in each spring
    w = c @ e  # Internal forces (stresses) in each spring
        
    return k, condition_number, V, S_matrix, f, u, eigen_values, e, w
        
        
if __name__ == "__main__":
    
    
    
    
    # Part 2 compare the results of my SVD vs the blackbox one
    
        # Test matrix
    A = np.array([[2, -1, 0], [1, 2, 1], [0, 1, 3]])

    # testing my svd
    U_custom, sigma_custom, V_custom, cond_custom, A_inv_custom, eigens_ATA_custom = svd_calc(A)

    # Run NumPy's SVD function
    U_np, s_np, V_np_T = np.linalg.svd(A)

    # Convert NumPy's singular values to a diagonal matrix to match your format
    sigma_np = np.zeros_like(A, dtype=float)
    np.fill_diagonal(sigma_np, s_np)

    # NumPy's condition number and inverse
    cond_np = np.linalg.cond(A)
    A_inv_np = np.linalg.pinv(A)  # Uses pseudo-inverse for robustness

    print("Custom SVD results:")
    print("U matrix:\n", U_custom)
    print("Singular values (Sigma):\n", sigma_custom)
    print("V matrix:\n", V_custom)
    print("Condition number:\n", cond_custom)
    print("Inverse matrix:\n", A_inv_custom)
    print("Eigens of A^T A:\n", eigens_ATA_custom)

    print("\nNumPy SVD results:")
    print("U matrix:\n", U_np)
    print("Singular values (Sigma):\n", sigma_np)
    print("V matrix:\n", V_np_T.T)
    print("Condition number:\n", cond_np)
    print("Inverse matrix:\n", A_inv_np)
    
    print("-----------------------------------------------------------------------")

    print("My results match well with the black box SVD function. You can tell because the condition numbers, eigenvalues, and U matrix (despite having the same numbers with different columns and signs) match. The reason the U matrices do not need to match exactly is due to the non-uniqueness that occurs in the U and V matrices. Non-uniqueness refers to the basis vectors that span the column and row spaces of A. A different choice of column or sign just changes the direction of the basis vectors and does not jeopardize the solution.")

    print("-----------------------------------------------------------------------")


    # Define the system parameters
    masses = np.array([1.0, 2.0, 1.5])         # Masses in kg
    spring_constants = np.array([100, 150, 150, 100])  # Spring constants in N/m
    springs = 4
    fixed_ends = 1

    # Call the spring_mass function with the example parameters
    k, condition_number, V, S, f, u, eigen_values, e, w = spring_mass(masses, springs, fixed_ends, spring_constants)
    print("-----------------------------------------------------------------------")

    # Display the results
    print("Stiffness matrix K:\n", k)
    print("Condition number of K:", condition_number)
    print("Singular values (S):\n", S)
    print("Force vector (f):\n", f)
    print("Displacement vector (u):\n", u)
    print("Elongation vector (e):\n", e)
    print("Internal force (stress) vector (w):\n", w)
    print("Eigenvalues of K:\n", eigen_values)
    
    k, condition_number, V, S_matrix, f, u, eigen_values, e, w = Spring_mass_with_user_inputs()
    print("-----------------------------------------------------------------------")
    # Display the results
    print("Stiffness matrix K:\n", k)
    print("Condition number of K:", condition_number)
    print("Singular values (S):\n", S)
    print("Force vector (f):\n", f)
    print("Displacement vector (u):\n", u)
    print("Elongation vector (e):\n", e)
    print("Internal force (stress) vector (w):\n", w)
    print("Eigenvalues of K:\n", eigen_values)