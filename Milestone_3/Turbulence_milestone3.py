import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt

# Simulation parameters
grid_size = 32  # Grid resolution (32x32x32)
domain_size = 1.0  # Size of the domain
kinematic_viscosity = 1 / 1600  # Viscosity constant
time_step = 0.01  # Timestep size
simulation_duration = 5.0  # Total time for simulation
initial_velocity_scale = 1.0  # Scaling factor for initial velocities

# Initialize FFT object
fft_resolution = (grid_size, grid_size, grid_size)
fft_solver = FFT(fft_resolution, engine='pocketfft')
spacing = domain_size / grid_size  # Grid cell size
# Construct wavevectors
k_vectors = (2 * np.pi * fft_solver.fftfreq.T / spacing).T
zero_mode = (k_vectors.T == np.zeros(3, dtype=int)).T.all(axis=0)
k_squared = np.sum(k_vectors ** 2, axis=0)

def initialize_velocity_field():
    random_vel_field = np.zeros((3,) + fft_solver.nb_fourier_grid_pts, dtype=complex)
    rng = np.random.default_rng()
    random_vel_field.real = rng.standard_normal(random_vel_field.shape)
    random_vel_field.imag = rng.standard_normal(random_vel_field.shape)

    factor = np.zeros_like(k_squared)
    factor[~zero_mode] = initial_velocity_scale * k_squared[~zero_mode] ** (-5 / 6)
    random_vel_field *= factor

    k_dot_velocity = np.sum(k_vectors * random_vel_field, axis=0)
    for i in range(3):
        # Prevent division by zero
        random_vel_field[i] -= (k_vectors[i] * k_dot_velocity) / np.where(k_squared == 0, 1, k_squared)

    random_vel_field[:, zero_mode] = 0

    return random_vel_field

def navier_stokes_rhs_spectrum(velocity_hat):
    velocity_real = np.array([fft_solver.ifft(velocity_hat[i]) for i in range(3)])

    nonlinear_term = np.array([
        fft_solver.fft(velocity_real[1] * velocity_real[2]),
        fft_solver.fft(velocity_real[2] * velocity_real[0]),
        fft_solver.fft(velocity_real[0] * velocity_real[1])
    ])

    rhs_term = -1j * np.cross(k_vectors, nonlinear_term, axis=0)
    rhs_term -= kinematic_viscosity * k_squared * velocity_hat

    k_dot_rhs_term = np.sum(k_vectors * rhs_term, axis=0)
    for i in range(3):
        rhs_term[i] -= (k_vectors[i] * k_dot_rhs_term) / np.where(k_squared == 0, 1, k_squared)

    return rhs_term

def rk4_integrate(velocity_hat, dt):
    k1 = navier_stokes_rhs_spectrum(velocity_hat)
    k2 = navier_stokes_rhs_spectrum(velocity_hat + 0.5 * dt * k1)
    k3 = navier_stokes_rhs_spectrum(velocity_hat + 0.5 * dt * k2)
    k4 = navier_stokes_rhs_spectrum(velocity_hat + dt * k3)
    return velocity_hat + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def enforce_forcing(velocity_hat):
    forcing_mask = (np.sqrt(k_squared) <= 2 * np.pi / domain_size)
    velocity_hat[:, forcing_mask] *= np.exp(kinematic_viscosity * k_squared[forcing_mask] * time_step)
    return velocity_hat

def normalize_vel_field(velocity_hat):
    total_energy = np.sum(np.abs(velocity_hat) ** 2)
    return velocity_hat * np.sqrt(1 / total_energy)

def compute_energy_dissipation_spectra(velocity_hat):
    energy_spec = np.zeros(grid_size // 2)
    dissipation_spec = np.zeros(grid_size // 2)

    for i in range(grid_size // 2):
        shell = (i <= np.sqrt(k_squared)) & (np.sqrt(k_squared) < i + 1)
        energy_spec[i] = 0.5 * np.sum(np.abs(velocity_hat[:, shell]) ** 2) / (grid_size ** 3)
        dissipation_spec[i] = 2 * kinematic_viscosity * np.sum(k_squared[shell] * np.abs(velocity_hat[:, shell]) ** 2) / (grid_size ** 3)

    return energy_spec, dissipation_spec

def save_simulation_plots(current_time, velocity_hat):
    velocity_real = np.array([fft_solver.ifft(velocity_hat[i]) for i in range(3)])

    # Plot velocity magnitude
    plt.figure(figsize=(10, 10))
    plt.imshow(np.sqrt(np.sum(velocity_real[:, :, :, grid_size // 2] ** 2, axis=0)), cmap='viridis')
    plt.colorbar()
    plt.title(f'Velocity Magnitude at t={current_time:.2f}', fontsize=14)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.savefig(f'velocity_magnitude_t{current_time:.2f}.png')
    plt.close()

    # Plot energy and dissipation spectra
    energy_spec, dissipation_spec = compute_energy_dissipation_spectra(velocity_hat)

    plt.figure(figsize=(12, 6))
    plt.loglog(range(1, grid_size // 2 + 1), energy_spec, label='Energy Spectrum', color='b')
    plt.loglog(range(1, grid_size // 2 + 1), dissipation_spec, label='Dissipation Spectrum', color='r')
    plt.loglog(range(1, grid_size // 2 + 1), np.array(range(1, grid_size // 2 + 1)) ** (-5 / 3), '--', label='k^(-5/3)', color='g')
    plt.xlabel('Wavenumber k', fontsize=12)
    plt.ylabel('Spectrum', fontsize=12)
    plt.legend()
    plt.title(f'Energy & Dissipation Spectra at t={current_time:.2f}', fontsize=14)
    plt.savefig(f'spectra_t{current_time:.2f}.png')
    plt.close()

# Set up initial velocity field
u_hat_initial = initialize_velocity_field()

# Time-stepping variables
current_time = 0
snapshot_times = [0, simulation_duration / 4, simulation_duration / 2, 3 * simulation_duration / 4, simulation_duration]
snapshot_index = 0

# Save initial state
save_simulation_plots(current_time, u_hat_initial)
snapshot_index += 1

# Time integration loop
while current_time < simulation_duration:
    u_hat_initial = rk4_integrate(u_hat_initial, time_step)
    u_hat_initial = enforce_forcing(u_hat_initial)
    u_hat_initial = normalize_vel_field(u_hat_initial)
    current_time += time_step

    if snapshot_index < len(snapshot_times) and current_time >= snapshot_times[snapshot_index]:
        save_simulation_plots(current_time, u_hat_initial)
        snapshot_index += 1

    print(f"Simulated Time: {current_time:.2f} / {simulation_duration:.2f}")

# Final save of simulation
if snapshot_index < len(snapshot_times):
    save_simulation_plots(current_time, u_hat_initial)

print("Simulation completed.")
