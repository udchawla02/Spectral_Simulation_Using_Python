import numpy as np
import muFFT as mfft
import matplotlib.pyplot as plt

# Initialize grid and FFT
grid_size = 64  # Number of grid points in each direction
domain_length = 2 * np.pi  # Size of the domain
viscosity = 0.01  # Kinematic viscosity

fft_transformer = mfft.FFT([grid_size, grid_size, grid_size])

# Calculate wavenumbers
wavenumbers = fft_transformer.fftfreq
kx, ky, kz = wavenumbers[0, :, :, :], wavenumbers[1, :, :, :], wavenumbers[2, :, :, :]
k_squared = kx ** 2 + ky ** 2 + kz ** 2
k_squared[0, 0, 0] = 1  # Prevent division by zero

def compute_rhs(time, u_hat):
    """Calculate the right-hand side of the Navier-Stokes equations in Fourier space"""
    u_real = fft_transformer.ifft(u_hat)

    # Compute nonlinear term in real space
    nonlinear_term = np.array([
        u_real[0] * np.gradient(u_real[0], axis=0) + u_real[1] * np.gradient(u_real[0], axis=1) + u_real[2] * np.gradient(u_real[0], axis=2),
        u_real[0] * np.gradient(u_real[1], axis=0) + u_real[1] * np.gradient(u_real[1], axis=1) + u_real[2] * np.gradient(u_real[1], axis=2),
        u_real[0] * np.gradient(u_real[2], axis=0) + u_real[1] * np.gradient(u_real[2], axis=1) + u_real[2] * np.gradient(u_real[2], axis=2)
    ])

    # Transform nonlinear term to Fourier space
    nonlinear_hat = fft_transformer.fft(nonlinear_term)

    # Ensure incompressibility
    P_k = 1 - (kx ** 2 + ky ** 2 + kz ** 2) / k_squared

    # Compute right-hand side
    rhs = -1j * (
            kx * P_k * nonlinear_hat[0] +
            ky * P_k * nonlinear_hat[1] +
            kz * P_k * nonlinear_hat[2]
    ) - viscosity * k_squared * u_hat

    return rhs

def runge_kutta4(func, time, y, delta_t):
    """Fourth-order Runge-Kutta method"""
    k1 = func(time, y)
    k2 = func(time + delta_t / 2, y + delta_t / 2 * k1)
    k3 = func(time + delta_t / 2, y + delta_t / 2 * k2)
    k4 = func(time + delta_t, y + delta_t * k3)
    return delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Initialize Taylor-Green vortex conditions
x_coords = np.linspace(0, domain_length, grid_size, endpoint=False)
y_coords = np.linspace(0, domain_length, grid_size, endpoint=False)
z_coords = np.linspace(0, domain_length, grid_size, endpoint=False)
X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

initial_conditions = np.array([
    np.sin(X) * np.cos(Y) * np.cos(Z),
    -np.cos(X) * np.sin(Y) * np.cos(Z),
    np.zeros_like(X)
])
u_hat = fft_transformer.fft(initial_conditions)

# Plot the initial velocity field
def plot_velocity(u, title):
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection='3d')
    magnitude = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    magnitude[magnitude == 0] = 1  # Avoid division by zero
    ax.quiver(X, Y, Z, u[0] / magnitude, u[1] / magnitude, u[2] / magnitude, length=0.1, normalize=True, color='c')
    ax.set_xlim([0, domain_length])
    ax.set_ylim([0, domain_length])
    ax.set_zlim([0, domain_length])
    ax.set_title(title)
    plt.show()

plot_velocity(initial_conditions, 'Initial Velocity Field')

# Time-stepping parameters
delta_t = 0.01
end_time = 1.0
steps = int(end_time / delta_t)

# Simulation loop
current_time = 0
energy_values = []

for step in range(steps):
    u_hat += runge_kutta4(compute_rhs, current_time, u_hat, delta_t)
    current_time += delta_t

    energy = np.sum(np.abs(u_hat) ** 2) / (2 * grid_size ** 3)
    energy_values.append(energy)

    if step % 10 == 0:
        print(f"Step {step}, t = {current_time:.3f}, Energy = {energy:.6f}")

# Convert final result back to real space
final_solution = fft_transformer.ifft(u_hat)

# Analytical solution for comparison
analytical_solution_hat = u_hat * np.exp(-2 * viscosity * k_squared * end_time)
analytical_solution = fft_transformer.ifft(analytical_solution_hat)

# Plot initial, final, and analytical velocity fields
def plot_comparison_fields(init_u, final_u, analytical_u):
    fig = plt.figure(figsize=(18, 6))

    norm_init = np.sqrt(init_u[0] ** 2 + init_u[1] ** 2 + init_u[2] ** 2)
    norm_final = np.sqrt(final_u[0] ** 2 + final_u[1] ** 2 + final_u[2] ** 2)
    norm_analytical = np.sqrt(analytical_u[0] ** 2 + analytical_u[1] ** 2 + analytical_u[2] ** 2)

    norm_init[norm_init == 0] = 1
    norm_final[norm_final == 0] = 1
    norm_analytical[norm_analytical == 0] = 1

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.quiver(X, Y, Z, init_u[0] / norm_init, init_u[1] / norm_init, init_u[2] / norm_init, length=0.1, color='r')
    ax1.set_xlim([0, domain_length])
    ax1.set_ylim([0, domain_length])
    ax1.set_zlim([0, domain_length])
    ax1.set_title('Initial Velocity Field')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.quiver(X, Y, Z, final_u[0] / norm_final, final_u[1] / norm_final, final_u[2] / norm_final, length=0.1, color='g')
    ax2.set_xlim([0, domain_length])
    ax2.set_ylim([0, domain_length])
    ax2.set_zlim([0, domain_length])
    ax2.set_title('Numerical Solution at Final Time')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.quiver(X, Y, Z, analytical_u[0] / norm_analytical, analytical_u[1] / norm_analytical, analytical_u[2] / norm_analytical, length=0.1, color='b')
    ax3.set_xlim([0, domain_length])
    ax3.set_ylim([0, domain_length])
    ax3.set_zlim([0, domain_length])
    ax3.set_title('Analytical Solution at Final Time')

    plt.show()

plot_comparison_fields(initial_conditions, final_solution, analytical_solution)

# Plot energy over time
plt.figure()
plt.plot(np.linspace(0, end_time, steps), energy_values, color='m')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy of the System Over Time')
plt.show()

print("Simulation complete")
