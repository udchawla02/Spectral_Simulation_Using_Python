import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt

# Simulation Parameters
GRID_DIM = 32  # 32x32x32 grid
DOMAIN_SIZE = 1.0  # Size of simulation domain
VISC = 1 / 1600  # Kinematic viscosity
STEP = 0.01  # Time step for simulation
FINAL_TIME = 5.0  # Total simulation time
INIT_VELOCITY_MAG = 1.0  # Amplitude for initial velocity

# FFT Initialization
dimensions = (GRID_DIM, GRID_DIM, GRID_DIM)
fft_engine = FFT(dimensions, engine='pocketfft')

# Grid Spacing
grid_spacing = DOMAIN_SIZE / GRID_DIM

# Wavevector Computation
wave_vectors = (2 * np.pi * fft_engine.fftfreq.T / grid_spacing).T
zero_wave_mask = (wave_vectors.T == np.zeros(3, dtype=int)).T.all(axis=0)
wave_vectors_sq = np.sum(wave_vectors ** 2, axis=0)

# Initialize Velocity Field Function
def initialize_velocity_field():
    velocity = np.zeros((3,) + fft_engine.nb_fourier_grid_pts, dtype=complex)
    rng = np.random.default_rng()
    velocity.real = rng.standard_normal(velocity.shape)
    velocity.imag = rng.standard_normal(velocity.shape)

    # Apply scaling to initialize velocity field
    scale = np.zeros_like(wave_vectors_sq)
    scale[np.logical_not(zero_wave_mask)] = INIT_VELOCITY_MAG * \
                                            wave_vectors_sq[np.logical_not(zero_wave_mask)] ** (-5 / 6)
    velocity *= scale

    # Enforce incompressibility
    k_dot_u = np.sum(wave_vectors * velocity, axis=0)
    for i in range(3):
        velocity[i] -= (wave_vectors[i] * k_dot_u) / np.where(wave_vectors_sq == 0, 1, wave_vectors_sq)

    velocity[:, zero_wave_mask] = 0
    return velocity

# Aliasing Correction (2/3 Rule) Application
def apply_2thirds_rule(data):
    filter_mask = np.ones(data.shape, dtype=bool)
    for i in range(3):
        filter_mask[:, fft_engine.nb_fourier_grid_pts[i] // 3:2 * fft_engine.nb_fourier_grid_pts[i] // 3] = False
    return data * filter_mask

# Navier-Stokes RHS Computation
def compute_navier_stokes_rhs(velocity_transformed, aliasing_correction=False):
    velocity_real = np.array([fft_engine.ifft(velocity_transformed[i]) for i in range(3)])

    if aliasing_correction:
        velocity_real = apply_2thirds_rule(velocity_real)

    nonlinear_term = np.array([
        fft_engine.fft(velocity_real[1] * velocity_real[2]),
        fft_engine.fft(velocity_real[2] * velocity_real[0]),
        fft_engine.fft(velocity_real[0] * velocity_real[1])
    ])

    if aliasing_correction:
        nonlinear_term = apply_2thirds_rule(nonlinear_term)

    # Compute Navier-Stokes RHS in Fourier space
    rhs = -1j * np.cross(wave_vectors, nonlinear_term, axis=0)
    rhs -= VISC * wave_vectors_sq * velocity_transformed

    # Enforce incompressibility
    k_dot_rhs = np.sum(wave_vectors * rhs, axis=0)
    for i in range(3):
        rhs[i] -= (wave_vectors[i] * k_dot_rhs) / np.where(wave_vectors_sq == 0, 1, wave_vectors_sq)

    return rhs

# Fourth-order Runge-Kutta Scheme
def runge_kutta_4_step(velocity_transformed, time_step, aliasing_correction=False):
    k1 = compute_navier_stokes_rhs(velocity_transformed, aliasing_correction)
    k2 = compute_navier_stokes_rhs(velocity_transformed + 0.5 * time_step * k1, aliasing_correction)
    k3 = compute_navier_stokes_rhs(velocity_transformed + 0.5 * time_step * k2, aliasing_correction)
    k4 = compute_navier_stokes_rhs(velocity_transformed + time_step * k3, aliasing_correction)
    return velocity_transformed + (time_step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Forcing Function for Low-Wavenumber Modes
def apply_external_forcing(velocity_transformed):
    low_wavenumber_mask = (np.sqrt(wave_vectors_sq) <= 2 * np.pi / DOMAIN_SIZE)
    velocity_transformed[:, low_wavenumber_mask] *= np.exp(VISC * wave_vectors_sq[low_wavenumber_mask] * STEP)
    return velocity_transformed

# Normalize Velocity Field to Maintain Energy
def normalize_velocity_energy(velocity_transformed):
    total_energy = np.sum(np.abs(velocity_transformed) ** 2)
    return velocity_transformed * np.sqrt(1 / total_energy)

# Spectrum Calculation for Energy and Dissipation
def calculate_spectra(velocity_transformed):
    energy_spectrum = np.zeros(GRID_DIM // 2)
    dissipation_spectrum = np.zeros(GRID_DIM // 2)

    for i in range(GRID_DIM // 2):
        shell = (i <= np.sqrt(wave_vectors_sq)) & (np.sqrt(wave_vectors_sq) < i + 1)
        energy_spectrum[i] = 0.5 * np.sum(np.abs(velocity_transformed[:, shell]) * 2) / (GRID_DIM * 3)
        dissipation_spectrum[i] = 2 * VISC * np.sum(wave_vectors_sq[shell] * np.abs(velocity_transformed[:, shell]) * 2) / (
                    GRID_DIM * 3)

    return energy_spectrum, dissipation_spectrum

# Plotting and Saving Results
def save_results_plot(time, velocity_transformed_no_correction, velocity_transformed_with_correction):
    # Compute spectra
    spectrum_no_correction, dissipation_no_correction = calculate_spectra(velocity_transformed_no_correction)
    spectrum_with_correction, dissipation_with_correction = calculate_spectra(velocity_transformed_with_correction)

    # Plot the energy and dissipation spectra
    plt.figure(figsize=(12, 6))
    plt.loglog(range(1, GRID_DIM // 2 + 1), spectrum_no_correction, label='Energy (Without Correction)', color='teal')
    plt.loglog(range(1, GRID_DIM // 2 + 1), spectrum_with_correction, label='Energy (With Correction)', color='coral')
    plt.loglog(range(1, GRID_DIM // 2 + 1), dissipation_no_correction, label='Dissipation (Without Correction)', color='lime')
    plt.loglog(range(1, GRID_DIM // 2 + 1), dissipation_with_correction, label='Dissipation (With Correction)', color='tomato')
    plt.loglog(range(1, GRID_DIM // 2 + 1), np.array(range(1, GRID_DIM // 2 + 1)) ** (-5 / 3), '--', label='k^(-5/3)', color='gray')
    plt.xlabel('Wavenumber (k)')
    plt.ylabel('Spectrum')
    plt.legend()
    plt.title(f'Spectra Analysis at t={time:.2f}')
    plt.savefig(f'spectra_analysis_t{time:.2f}.png')
    plt.close()

    # Visualize velocity magnitude field
    real_velocity_no_correction = np.array([fft_engine.ifft(velocity_transformed_no_correction[i]) for i in range(3)])
    real_velocity_with_correction = np.array([fft_engine.ifft(velocity_transformed_with_correction[i]) for i in range(3)])

    plt.figure(figsize=(10, 10))
    plt.imshow(np.sqrt(np.sum(real_velocity_no_correction[:, :, :, GRID_DIM // 2] ** 2, axis=0)), cmap='Greens')
    plt.colorbar()
    plt.title(f'Velocity Magnitude (Without Correction) at t={time:.2f}')
    plt.savefig(f'velocity_field_no_correction_t{time:.2f}.png')
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(np.sqrt(np.sum(real_velocity_with_correction[:, :, :, GRID_DIM // 2] ** 2, axis=0)), cmap='Reds')
    plt.colorbar()
    plt.title(f'Velocity Magnitude (With Correction) at t={time:.2f}')
    plt.savefig(f'velocity_field_with_correction_t{time:.2f}.png')
    plt.close()

# Initialize velocity field
velocity_transformed_no_correction = initialize_velocity_field()
velocity_transformed_with_correction = velocity_transformed_no_correction.copy()

# Simulation Loop
current_time = 0
plot_times = [0, FINAL_TIME / 4, FINAL_TIME / 2, 3 * FINAL_TIME / 4, FINAL_TIME]
next_plot_time_index = 0

# Save initial plots
save_results_plot(current_time, velocity_transformed_no_correction, velocity_transformed_with_correction)
next_plot_time_index += 1

while current_time < FINAL_TIME:
    velocity_transformed_no_correction = runge_kutta_4_step(velocity_transformed_no_correction, STEP, aliasing_correction=False)
    velocity_transformed_with_correction = runge_kutta_4_step(velocity_transformed_with_correction, STEP, aliasing_correction=True)

    velocity_transformed_no_correction = apply_external_forcing(velocity_transformed_no_correction)
    velocity_transformed_with_correction = apply_external_forcing(velocity_transformed_with_correction)

    velocity_transformed_no_correction = normalize_velocity_energy(velocity_transformed_no_correction)
    velocity_transformed_with_correction = normalize_velocity_energy(velocity_transformed_with_correction)

    current_time += STEP

    if next_plot_time_index < len(plot_times) and current_time >= plot_times[next_plot_time_index]:
        save_results_plot(current_time, velocity_transformed_no_correction, velocity_transformed_with_correction)
        next_plot_time_index += 1

    print(f"Simulated Time: {current_time:.2f} / {FINAL_TIME}")

# Final Save
if next_plot_time_index < len(plot_times):
    save_results_plot(current_time, velocity_transformed_no_correction, velocity_transformed_with_correction)

print("Simulation Complete.")