import numpy as np
import matplotlib.pyplot as plt

# Define the right-hand side function f(x, t) and exact solution
f = lambda x, t: (np.pi**2 - 1) * np.exp(-t) * np.sin(np.pi * x)
u_exact = lambda x, t: np.exp(-t) * np.sin(np.pi * x)

# Function to create uniform grid and connectivity
def create_uniform_grid_and_connectivity(N, x_left, x_right):
    Ne = N - 1
    h = (x_right - x_left) / Ne
    x = np.linspace(x_left, x_right, N)
    iee = np.array([[i, i + 1] for i in range(Ne)])
    return x, iee, h

# Define basis functions
def define_basis_functions():
    phi_1 = lambda xi: 0.5 * (1 - xi)
    phi_2 = lambda xi: 0.5 * (1 + xi)
    dphi_1_dxi = lambda xi: -0.5
    dphi_2_dxi = lambda xi: 0.5
    return phi_1, phi_2, dphi_1_dxi, dphi_2_dxi

# Compute the global mass matrix
def compute_mass_matrix(N, iee, phi_1, phi_2, h):
    M_global = np.zeros((N, N))
    gauss_points = [-np.sqrt(1/3), np.sqrt(1/3)]
    gauss_weights = [1, 1]
    for elem in iee:
        m_local = np.zeros((2, 2))
        for gp, gw in zip(gauss_points, gauss_weights):
            phi = [phi_1(gp), phi_2(gp)]
            for l in range(2):
                for m in range(2):
                    m_local[l, m] += gw * phi[l] * phi[m] * h / 2
        for l in range(2):
            for m in range(2):
                M_global[elem[l], elem[m]] += m_local[l, m]
    return M_global

# Compute local element matrices
def compute_local_element_matrices(h, phi_1, phi_2, dphi_1_dxi, dphi_2_dxi, f, x_element, t):
    gauss_points = [-np.sqrt(1/3), np.sqrt(1/3)]
    gauss_weights = [1, 1]
    k_local = np.zeros((2, 2))
    f_local = np.zeros(2)
    for gp, gw in zip(gauss_points, gauss_weights):
        phi = [phi_1(gp), phi_2(gp)]
        dphi_dx = [dphi_1_dxi(gp) * 2 / h, dphi_2_dxi(gp) * 2 / h]
        x = 0.5 * (1 - gp) * x_element[0] + 0.5 * (1 + gp) * x_element[1]
        f_val = f(x, t)
        for l in range(2):
            f_local[l] += gw * f_val * phi[l] * h / 2
        for l in range(2):
            for m in range(2):
                k_local[l, m] += gw * dphi_dx[l] * dphi_dx[m] * h / 2
    return k_local, f_local

# Assemble global matrices
def assemble_global_matrices(N, iee, k_local_all, f_local_all):
    K_global = np.zeros((N, N))
    F_global = np.zeros(N)
    for k, elem in enumerate(iee):
        for l in range(2):
            F_global[elem[l]] += f_local_all[k][l]
            for m in range(2):
                K_global[elem[l], elem[m]] += k_local_all[k][l, m]
    return K_global, F_global
# Assemble RHS vector
def assemble_rhs(f, iee, x, h, t, gp, gw):
    F_global = np.zeros(len(x))
    for elem in iee:
        flocal = np.zeros(2)
        for l in range(2):
            flocal[l] = sum(
                f(0.5 * (1 - gp[j]) * x[elem[0]] + 0.5 * (1 + gp[j]) * x[elem[1]], t) * (1 + (-1)**l * gp[j]) / 2 * gw[j]
                for j in range(len(gp))
            ) * h / 2
        for l in range(2):
            F_global[elem[l]] += flocal[l]
    return F_global
# Apply Dirichlet boundary conditions with explicit enforcement of boundary values
def apply_dirichlet_bc(matrix, vector, boundary_nodes, N):
    """
    Modify the global matrix and RHS vector to enforce Dirichlet boundary conditions.
    """
    for node in boundary_nodes:
        for j in range(N):
            matrix[node, j] = 0  # Zero out the row
            matrix[j, node] = 0  # Zero out the column
        matrix[node, node] = 1  # Set diagonal to 1
        vector[node] = 0       # Set RHS value to 0 for boundary nodes
    return matrix, vector

# Solve using Forward Euler with explicit boundary enforcement
def solve_forward_euler(K_global, M_global, u_initial, dt, T0, Tf, iee, x, h, f, gp, gw, boundary_nodes):
    """
    Solve the heat equation using Forward Euler, ensuring boundary conditions are enforced.
    """
    u = np.copy(u_initial)
    M_inv = np.linalg.inv(M_global)
    nt = int((Tf - T0) / dt)

    for n in range(nt):
        t = T0 + n * dt

        # Assemble the RHS vector
        F_global = assemble_rhs(f, iee, x, h, t, gp, gw)

        # Forward Euler update
        rhs = -K_global @ u + F_global
        u = u + dt * (M_inv @ rhs)

        # Enforce boundary conditions
        for node in boundary_nodes:
            u[node] = 0

    return u

# Solve using Backward Euler with explicit boundary enforcement
def solve_backward_euler(K_global, M_global, u_initial, dt, T0, Tf, iee, x, h, f, gp, gw, boundary_nodes):
    """
    Solve the heat equation using Backward Euler, solving explicitly using the given equation.
    """
    u = np.copy(u_initial)
    nt = int((Tf - T0) / dt)
    B = (1 / dt) * M_global + K_global  # B matrix as shown in the equation
    B_inv = np.linalg.inv(B)  # Precompute B^-1

    for n in range(nt):
        t = T0 + n * dt

        # Assemble the RHS vector
        F_global = assemble_rhs(f, iee, x, h, t, gp, gw)

        # Explicit computation of u^{n+1} using the given formula
        rhs = (1 / dt) * M_global @ u + F_global
        u_next = (1 / dt) * (B_inv @ M_global @ u) + B_inv @ F_global

        # Enforce boundary conditions
        for node in boundary_nodes:
            u_next[node] = 0

        u = u_next

    return u
# Updated main function to overlay exact solution with each Euler method
def main():
    N = 11
    x_left, x_right = 0, 1
    T0, Tf = 0, 1
    dt = 1 / 551

    # 2d quad points
    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    gw = [1, 1]

    x, iee, h = create_uniform_grid_and_connectivity(N, x_left, x_right)
    phi_1, phi_2, dphi_1_dxi, dphi_2_dxi = define_basis_functions()

    M_global = compute_mass_matrix(N, iee, phi_1, phi_2, h)
    k_local_all, f_local_all = zip(*[
        compute_local_element_matrices(h, phi_1, phi_2, dphi_1_dxi, dphi_2_dxi, f, [x[i], x[i + 1]], T0)
        for i in range(len(iee))
    ])
    K_global, F_global = assemble_global_matrices(N, iee, k_local_all, f_local_all)

    boundary_nodes = [0, N - 1]
    K_global, F_global = apply_dirichlet_bc(K_global, F_global, boundary_nodes, N)
    M_global, _ = apply_dirichlet_bc(M_global, F_global, boundary_nodes, N)

    u_initial = np.sin(np.pi * np.array(x))

    u_forward = solve_forward_euler(K_global, M_global, u_initial, dt, T0, Tf, iee, x, h, f, gp, gw, boundary_nodes)
    u_backward = solve_backward_euler(K_global, M_global, u_initial, dt, T0, Tf, iee, x, h, f, gp, gw, boundary_nodes)
    u_exact_vals = u_exact(x, Tf)

    # Plot Forward Euler solution with exact solution
    plt.figure()
    plt.plot(x, u_forward, label="Forward Euler")
    plt.plot(x, u_exact_vals,color = 'r', label="Exact Solution")
    plt.title("Forward Euler vs Exact Solution")
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Backward Euler solution with exact solution
    plt.figure()
    plt.plot(x, u_backward, label="Backward Euler")
    plt.plot(x, u_exact_vals, color = 'r', label="Exact Solution")
    plt.title("Backward Euler vs Exact Solution")
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
