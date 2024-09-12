import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import muFFT
import muGrid

class FFTProcessor:
    def __init__(self, grid_points, method='pocketfft'):
        """Initialize the FFT processor with grid points and method."""
        self.grid_points = grid_points
        self.method = method
        self.comm = muGrid.Communicator()
        self.fft = muFFT.FFT(grid_points, engine=self.method, communicator=self.comm)

    def compute_forward(self, real_space_field):
        """Perform forward FFT to convert real-space field to Fourier space."""
        real_field = self.fft.real_space_field('real_field')
        real_field.p = real_space_field
        fourier_field = self.fft.fourier_space_field('fourier_field')
        self.fft.fft(real_field, fourier_field)
        return fourier_field.p

    def compute_backward(self, fourier_space_field):
        """Perform inverse FFT to convert Fourier-space field to real space."""
        fourier_field = self.fft.fourier_space_field('fourier_field')
        fourier_field.p = fourier_space_field
        real_field = self.fft.real_space_field('real_field')
        self.fft.ifft(fourier_field, real_field)
        return real_field.p


def calculate_curl(vector_field):
    """Compute the curl of a 3D vector field using FFT."""
    grid_dims = vector_field.shape[:-1]
    fft_processor = FFTProcessor(grid_dims)

    # Transform each component of the vector field
    field_shape = fft_processor.compute_forward(vector_field[..., 0]).shape
    fourier_components = np.zeros(field_shape + (3,), dtype=np.complex128)

    for i in range(3):
        component = vector_field[..., i]
        fourier_components[..., i] = fft_processor.compute_forward(component)

    # Create wave number vectors
    kx = np.fft.fftfreq(field_shape[0]) * 2j * np.pi
    ky = np.fft.fftfreq(field_shape[1]) * 2j * np.pi
    kz = np.fft.fftfreq(field_shape[2]) * 2j * np.pi
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')

    # Calculate curl in Fourier space
    curl_fourier = np.zeros_like(fourier_components, dtype=np.complex128)
    curl_fourier[..., 0] = ky * fourier_components[..., 2] - kz * fourier_components[..., 1]
    curl_fourier[..., 1] = kz * fourier_components[..., 0] - kx * fourier_components[..., 2]
    curl_fourier[..., 2] = kx * fourier_components[..., 1] - ky * fourier_components[..., 0]

    # Inverse FFT to get the curl in real space
    real_curl_shape = fft_processor.compute_backward(curl_fourier[..., 0]).shape
    real_curl = np.zeros(real_curl_shape + (3,), dtype=np.float64)

    for i in range(3):
        real_curl[..., i] = fft_processor.compute_backward(curl_fourier[..., i]).real

    return real_curl


def test_constant_field_curl():
    """Test if the curl of a constant field is zero."""
    grid_points = (32, 32, 2)
    fft_processor = FFTProcessor(grid_points)

    constant_field = np.ones([*grid_points, 3])
    curl_result = calculate_curl(constant_field)
    np.testing.assert_allclose(curl_result, 0, atol=1e-10)
    print("Test passed: Constant field results in a zero curl.")


def create_and_curl_vector_field():
    """Generate a vector field and compute its curl."""
    grid_points = (32, 32, 2)
    fft_processor = FFTProcessor(grid_points)

    # Create a grid of coordinates
    coords = np.array(np.meshgrid(
        np.linspace(0, 1, grid_points[0]),
        np.linspace(0, 1, grid_points[1]),
        np.linspace(0, 1, grid_points[2]),
        indexing='ij'
    ))

    # Define a vector field as a cross product
    direction = np.array([0, 0, 1])
    vector_field = np.zeros(coords.shape[1:] + (3,))
    vector_field[..., 0] = direction[1] * (coords[2] - 0.5) - direction[2] * (coords[1] - 0.5)
    vector_field[..., 1] = direction[2] * (coords[0] - 0.5) - direction[0] * (coords[2] - 0.5)
    vector_field[..., 2] = direction[0] * (coords[1] - 0.5) - direction[1] * (coords[0] - 0.5)

    curl_result = calculate_curl(vector_field)
    return coords, vector_field, curl_result


def visualize_field_and_curl(coords, vector_field, curl_result):
    """Visualize the vector field and its curl."""
    fig = plt.figure(figsize=(14, 6))

    # Plot the vector field
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.quiver(coords[0], coords[1], coords[2], vector_field[..., 0], vector_field[..., 1], vector_field[..., 2],
               length=0.1, normalize=True, color='green', linewidth=0.5)
    ax1.set_title('Vector Field')

    # Plot the curl of the vector field
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.quiver(coords[0], coords[1], coords[2], curl_result[..., 0], curl_result[..., 1], curl_result[..., 2],
               length=0.1, normalize=True, color='orange', linewidth=0.5)
    ax2.set_title('Curl of the Vector Field')

    plt.show()


def display_curl_samples(curl_result):
    """Print sample values of the curl."""
    print("Sample curl values at various grid points:")
    for i in range(0, curl_result.shape[0], max(1, curl_result.shape[0] // 5)):
        for j in range(0, curl_result.shape[1], max(1, curl_result.shape[1] // 5)):
            for k in range(0, curl_result.shape[2], max(1, curl_result.shape[2] // 2)):
                print(f"Curl at ({i}, {j}, {k}): {curl_result[i, j, k]}")


def test_nonzero_curl():
    """Test a non-zero curl vector field and visualize the results."""
    print("Testing non-zero curl field")
    grid_points = (32, 32, 2)

    # Generate the vector field
    direction = np.array([0, 0, 1])
    coords = np.array(np.meshgrid(
        np.linspace(0, 1, grid_points[0]),
        np.linspace(0, 1, grid_points[1]),
        np.linspace(0, 1, grid_points[2]),
        indexing='ij'
    ))
    vector_field = np.cross(direction, coords - 0.5, axis=0)
    vector_field = np.moveaxis(vector_field, 0, -1)

    # Compute the curl
    curl_result = calculate_curl(vector_field)

    # Visualize the vector field and its curl
    visualize_field_and_curl(coords, vector_field, curl_result)

    # Print curl values and mean curl
    display_curl_samples(curl_result)
    mean_curl = np.mean(curl_result, axis=(0, 1, 2))
    print(f"Mean curl: {mean_curl}")


# Run tests
test_constant_field_curl()
test_nonzero_curl()
