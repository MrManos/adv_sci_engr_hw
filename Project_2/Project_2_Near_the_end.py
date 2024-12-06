import numpy as np
import matplotlib.pyplot as plt

def create_grid_and_connectivity(N, x_left, x_right):
    # Number of elements
    Ne = N - 1
    # Step size
    h = (x_right - x_left) / (N - 1)
    # Node positions
    x = [x_left + i * h for i in range(N)]
    # Connectivity map (assuming linear elements)
    iee = [[i, i + 1] for i in range(Ne)]
    return x, iee, h

def define_basis_functions():
    # Basis functions in parent space [-1, 1]
    phi_1 = lambda xi: 0.5 * (1 - xi)
    phi_2 = lambda xi: 0.5 * (1 + xi)
    # Derivatives of basis functions in parent space
    dphi_1_dxi = lambda xi: -0.5
    dphi_2_dxi = lambda xi: 0.5
    return phi_1, phi_2, dphi_1_dxi, dphi_2_dxi

def compute_mass_matrix(N, iee, phi_1, phi_2, h):
    M_global = np.zeros((N, N))
    gauss_points = [-np.sqrt(1/3), np.sqrt(1/3)]
    gauss_weights = [1, 1]
    for k, element in enumerate(iee):
        m_local = [[0, 0], [0, 0]]
        for gp, gw in zip(gauss_points, gauss_weights):
            phi = [phi_1(gp), phi_2(gp)]
            for l in range(2):
                for m in range(2):
                    m_local[l][m] += gw * phi[l] * phi[m] * h / 2
        for l in range(2):
            for m in range(2):
                global_node_l = element[l]
                global_node_m = element[m]
                M_global[global_node_l, global_node_m] += m_local[l][m]
    return M_global

def compute_local_element_matrices(h, phi_1, phi_2, dphi_1_dxi, dphi_2_dxi, f, x_element, t):
    gauss_points = [-np.sqrt(1/3), np.sqrt(1/3)]
    gauss_weights = [1, 1]
    k_local = [[0, 0], [0, 0]]
    f_local = [0, 0]
    for gp, gw in zip(gauss_points, gauss_weights):
        phi = [phi_1(gp), phi_2(gp)]
        dphi_dx = [dphi_1_dxi(gp) * 2 / h, dphi_2_dxi(gp) * 2 / h]
        x = 0.5 * (1 - gp) * x_element[0] + 0.5 * (1 + gp) * x_element[1]
        f_val = f(x, t)  # Pass time t to the forcing function
        for l in range(2):
            f_local[l] += gw * f_val * phi[l] * h / 2
        for l in range(2):
            for m in range(2):
                k_local[l][m] += gw * dphi_dx[l] * dphi_dx[m] * h / 2
    return k_local, f_local


def assemble_global_matrices(N, iee, k_local_all, f_local_all):
    # Initialize global matrices
    K_global = np.zeros((N, N))  # Global stiffness matrix
    F_global = np.zeros(N)       # Global RHS vector

    # Loop through each element
    for k, element in enumerate(iee):
        # Local-to-global node mapping
        global_nodes = element

        # Add local contributions to global matrices
        for l in range(2):
            F_global[global_nodes[l]] += f_local_all[k][l]
            for m in range(2):
                K_global[global_nodes[l], global_nodes[m]] += k_local_all[k][l][m]

    return K_global, F_global

def apply_natural_boundary_conditions(F_global, boundary_conditions, phi_1, phi_2, x_left, x_right):
    """
    Apply natural boundary conditions to the global force vector.
    """
    if "natural_left" in boundary_conditions:
        u_prime_left = boundary_conditions["natural_left"]
        F_global[0] += -u_prime_left * phi_1(x_left)

    if "natural_right" in boundary_conditions:
        u_prime_right = boundary_conditions["natural_right"]
        F_global[-1] += -u_prime_right * phi_2(x_right)
    
    return F_global

def apply_dirichlet_boundary_conditions(K_global, F_global, boundary_conditions, N):
    """
    Apply Dirichlet boundary conditions by modifying the global stiffness matrix and force vector.
    """
    for i in range(N):
        if i == 0 and "dirichlet_left" in boundary_conditions:
            u_left = boundary_conditions["dirichlet_left"]
            # Apply left Dirichlet condition
            for j in range(N):
                if i != j:
                    F_global[j] -= K_global[j][i] * u_left
                    K_global[j][i] = 0
                    K_global[i][j] = 0
            F_global[i] = u_left
            K_global[i][i] = 1

        elif i == N - 1 and "dirichlet_right" in boundary_conditions:
            u_right = boundary_conditions["dirichlet_right"]
            # Apply right Dirichlet condition
            for j in range(N):
                if i != j:
                    F_global[j] -= K_global[j][i] * u_right
                    K_global[j][i] = 0
                    K_global[i][j] = 0
            F_global[i] = u_right
            K_global[i][i] = 1

    return K_global, F_global

import numpy as np

def solve_heat_equation_backward_euler(K_global, M_global, F_global, u_initial, dt, T0, Tf, iee, f, phi_1, phi_2, h):
    """
    Solve the 1D heat equation using the implicit backward Euler method.

    Parameters:
    - K_global: Global stiffness matrix
    - M_global: Global mass matrix
    - F_global: Global force vector
    - u_initial: Initial condition vector
    - dt: Time step size
    - T0: Initial time
    - Tf: Final time
    - iee: Element connectivity map
    - f: External forcing function
    - phi_1, phi_2: Basis functions
    - h: Spatial step size

    Returns:
    - u: Solution at each time step
    """
    # Calculate M^-1 (Inverse of the mass matrix)
    M_inv = np.linalg.inv(M_global)

    # Number of time steps
    nt = int((Tf - T0) / dt)

    # Initial condition
    u = np.copy(u_initial)
    ctime = T0  # Current time

    # Time-stepping loop
    for n in range(nt):
        ctime = T0 + n * dt

        # Build time-dependent RHS vector
        F_time_dependent = np.zeros_like(F_global)
        for k, element in enumerate(iee):
            # Local element calculation
            x_element = [element[0] * h, element[1] * h]
            f_local = [0, 0]
            
            # Gaussian quadrature points and weights for [-1, 1]
            gauss_points = [-np.sqrt(1/3), np.sqrt(1/3)]
            gauss_weights = [1, 1]
            for gp, gw in zip(gauss_points, gauss_weights):
                # Evaluate basis functions at quadrature point
                phi = [phi_1(gp), phi_2(gp)]
                x = 0.5 * (1 - gp) * x_element[0] + 0.5 * (1 + gp) * x_element[1]
                f_val = f(x, ctime)

                for l in range(2):
                    f_local[l] += gw * f_val * phi[l] * h / 2

            # Finite element assembly
            for l in range(2):
                global_node = iee[k][l]
                F_time_dependent[global_node] += f_local[l]

        # Update solution: u^n+1 = u^n + dt * M^-1 * (-K*u^n + F^n)
        rhs = -np.dot(K_global, u) + F_time_dependent
        u = u + dt * np.dot(M_inv, rhs)

    return u




def solve_heat_equation_forward_euler(K_global, M_global, F_global, u_initial, dt, T0, Tf):
    """
    Solve the 1D heat equation using the forward Euler method.
    """
    nt = int((Tf - T0) / dt)  # Number of time steps
    u = np.copy(u_initial)  # Initial condition

    # Precompute M^-1
    M_inv = np.linalg.inv(M_global)

    # Time-stepping loop
    for n in range(nt):
        # Compute the RHS
        rhs = -np.dot(K_global, u) + F_global
        # Update the solution
        u = u + dt * np.dot(M_inv, rhs)

    return u

def main():
    # Problem parameters
    N = 11  # Number of nodes
    x_left, x_right = 0, 1
    T0, Tf = 0, 1
    dt_initial = 1 / 551  # Initial time step
    dt_values = [dt_initial, dt_initial * 2, dt_initial * 5]  # Different dt values for analysis

    # Generate grid and connectivity
    x, iee, h = create_grid_and_connectivity(N, x_left, x_right)

    # Initial condition: u(x, 0) = sin(πx)
    u_initial = np.sin(np.pi * np.array(x))

    # Boundary conditions
    boundary_conditions = {"dirichlet_left": 0, "dirichlet_right": 0}

    # Forcing function: f(x, t) = (π² - 1)e^(-t) * sin(πx)
    def forcing_function(x, t):
        return (np.pi**2 - 1) * np.exp(-t) * np.sin(np.pi * x)

    # Define basis functions
    phi_1, phi_2, dphi_1_dxi, dphi_2_dxi = define_basis_functions()

    # Compute global mass matrix
    M_global = compute_mass_matrix(N, iee, phi_1, phi_2, h)

    # Compute stiffness and force matrices
    k_local_all, f_local_all = zip(*[
        compute_local_element_matrices(h, phi_1, phi_2, dphi_1_dxi, dphi_2_dxi, forcing_function, [x[i], x[i + 1]], T0)
        for i in range(len(iee))
    ])
    K_global, F_global = assemble_global_matrices(N, iee, k_local_all, f_local_all)

    # Apply Dirichlet boundary conditions
    K_global, F_global = apply_dirichlet_boundary_conditions(K_global, F_global, boundary_conditions, N)

    # ---- Solve with Forward Euler ----
    plt.figure()
    for dt in dt_values:
        u_forward = solve_heat_equation_forward_euler(K_global, M_global, F_global, u_initial, dt, T0, Tf)
        plt.plot(x, u_forward, label=f"Forward Euler (dt={dt:.4f})")
    plt.xlabel("x")
    plt.ylabel("u(x, t=Tf)")
    plt.title("Forward Euler Solutions")
    plt.legend()
    plt.grid()
    plt.show()

    # Analysis of instability for Forward Euler
    print("Analyzing Forward Euler instability:")
    for dt in [dt_initial, dt_initial * 2, dt_initial * 5]:
        try:
            u_forward = solve_heat_equation_forward_euler(K_global, M_global, F_global, u_initial, dt, T0, Tf)
            print(f"Stable solution with dt = {dt:.4f}")
        except np.linalg.LinAlgError:
            print(f"Instability occurred for dt = {dt:.4f}")

    # ---- Solve with Backward Euler ----
    plt.figure()
    for dt in [h / 2, h, 2 * h]:  # Test dt < h, dt = h, and dt > h
        u_backward = solve_heat_equation_backward_euler(
            K_global, M_global, F_global, u_initial, dt, T0, Tf, iee, forcing_function, phi_1, phi_2, h
        )
        plt.plot(x, u_backward, label=f"Backward Euler (dt={dt:.4f})")
    plt.xlabel("x")
    plt.ylabel("u(x, t=Tf)")
    plt.title("Backward Euler Solutions")
    plt.legend()
    plt.grid()
    plt.show()

    # Summary of observations for Backward Euler
    print("\nBackward Euler Analysis:")
    print("For dt >= h, Backward Euler remains stable but introduces smoothing.")
    print("Numerical diffusion increases as dt grows larger.")

    # ---- Solution comparison as N decreases ----
    plt.figure()
    for N in [5, 11, 21]:  # Test with different mesh resolutions
        x, iee, h = create_grid_and_connectivity(N, x_left, x_right)
        u_initial = np.sin(np.pi * np.array(x))
        M_global = compute_mass_matrix(N, iee, phi_1, phi_2, h)
        k_local_all, f_local_all = zip(*[
            compute_local_element_matrices(h, phi_1, phi_2, dphi_1_dxi, dphi_2_dxi, forcing_function, [x[i], x[i + 1]], T0)
            for i in range(len(iee))
        ])
        K_global, F_global = assemble_global_matrices(N, iee, k_local_all, f_local_all)
        K_global, F_global = apply_dirichlet_boundary_conditions(K_global, F_global, boundary_conditions, N)

        u_forward = solve_heat_equation_forward_euler(K_global, M_global, F_global, u_initial, dt_initial, T0, Tf)
        plt.plot(x, u_forward, label=f"N={N}")
    plt.xlabel("x")
    plt.ylabel("u(x, t=Tf)")
    plt.title("Forward Euler Solution for Different N")
    plt.legend()
    plt.grid()
    plt.show()

    print("\nMesh Refinement Analysis:")
    print("As N decreases, the solution becomes less accurate due to coarser spatial resolution.")
    print("A larger N captures more detail but increases computational cost.")


if __name__ == "__main__":
    main()