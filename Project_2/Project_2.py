import numpy as np
import matplotlib.pyplot as plt

# Problem setup
L = 1.0  # Length of the domain
N = 11   # Number of nodes
x = np.linspace(0, L, N)
dx = x[1] - x[0]  # Spatial step size

# Time discretization
dt = 1 / 551  # Given time step
T_end = 1.0  # Final time
time_steps = int(T_end / dt)

# Function f(x, t)
def f(x, t):
    return (np.pi**2 - 1) * np.exp(-t) * np.sin(np.pi * x)

# Initialize global matrices
M = np.zeros((N, N))
K = np.zeros((N, N))

# Elemental matrices (local)
def elemental_matrices(dx):
    # Mass matrix (M_e)
    M_e = (dx / 6) * np.array([[2, 1], [1, 2]])
    # Stiffness matrix (K_e)
    K_e = (1 / dx) * np.array([[1, -1], [-1, 1]])
    return M_e, K_e

# Assemble global matrices
M_e, K_e = elemental_matrices(dx)
for e in range(N - 1):
    M[e:e+2, e:e+2] += M_e
    K[e:e+2, e:e+2] += K_e

# Apply boundary conditions to global matrices
M[0, :] = M[-1, :] = 0
M[0, 0] = M[-1, -1] = 1
K[0, :] = K[-1, :] = 0
K[0, 0] = K[-1, -1] = 1

# Initial condition: u(x, 0) = sin(pi * x)
u = np.sin(np.pi * x)
u_new = np.copy(u)

# Initialize array to store solutions for visualization
solutions_forward = [np.copy(u)]
solutions_backward = [np.copy(u)]

# Time integration: Forward Euler
for n in range(time_steps):
    F_current = f(x, n * dt)  # Evaluate f(x, t) at current time
    F_current[0] = F_current[-1] = 0  # Enforce boundary conditions on F
    u_new[1:-1] = u[1:-1] + dt * np.linalg.solve(M[1:-1, 1:-1], 
                       F_current[1:-1] - K[1:-1, 1:-1] @ u[1:-1])
    u = np.copy(u_new)
    solutions_forward.append(np.copy(u))

# Reset initial condition for Backward Euler
u = np.sin(np.pi * x)
u_new = np.copy(u)

# Time integration: Backward Euler
for n in range(time_steps):
    F_current = f(x, (n + 1) * dt)  # Evaluate f(x, t) at next time step
    F_current[0] = F_current[-1] = 0  # Enforce boundary conditions on F
    A = M + dt * K
    b = M @ u + dt * F_current
    u_new[1:-1] = np.linalg.solve(A[1:-1, 1:-1], b[1:-1])
    u = np.copy(u_new)
    solutions_backward.append(np.copy(u))

# Plot the results
times_to_plot = [0, int(time_steps / 4), int(time_steps / 2), time_steps - 1]
plt.figure(figsize=(10, 6))

# Forward Euler results
for t in times_to_plot:
    plt.plot(x, solutions_forward[t], label=f'Forward Euler t={t*dt:.2f}', linestyle='--')

# Backward Euler results
for t in times_to_plot:
    plt.plot(x, solutions_backward[t], label=f'Backward Euler t={t*dt:.2f}')

plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Comparison of Forward and Backward Euler Methods')
plt.legend()
plt.grid()
plt.show()
