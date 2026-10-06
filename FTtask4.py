from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np

image = io.imread("vehicle_type_recognition/hatchback/PIC_1.jpg")
gray_image = color.rgb2gray(image)

fourier_image = np.fft.fft2(gray_image)
shifted_fourier_image = np.fft.fftshift(fourier_image)
magnitude = np.abs(shifted_fourier_image)
magnitude_spectrum = np.log1p(magnitude)

rows, cols = gray_image.shape 
# Finner størrelsen på bildet og finner midten av bildet
center_row = rows // 2
center_col = cols // 2

radius = 50 # hvor stort område rundt sentrum vi skal beholde
mask = np.zeros((rows, cols)) # lager ny matrise like stor som bildet, fylt med nuller

# Lager en maske som beholder et sirkulært område rundt sentrum av bildet
y, x = np.ogrid[:rows, :cols]

distance_from_center = np.sqrt(
    (x - center_col)**2 + (y - center_row)**2
)

# Oppgave 4
# viser bare 10% av koeffisientene i frekvensbildet
percentage = 0.1

total_coefficients = magnitude.size
coefficients_to_keep = int(total_coefficients * percentage)

threshold = np.partition(
    magnitude.flatten(),
    -coefficients_to_keep
)[-coefficients_to_keep]




compression_mask = magnitude >= threshold
compressed_fourier = shifted_fourier_image * compression_mask

unshifted_compressed = np.fft.ifftshift(compressed_fourier)

reconstructed_image = np.fft.ifft2(unshifted_compressed)
reconstructed_image = np.real(reconstructed_image)

# Vise det rekonstruerte bildet med 10% av koeffisientene
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(gray_image, cmap="gray")
axes[0].set_title("Original image")
axes[0].axis("off")

axes[1].imshow(reconstructed_image, cmap="gray")
axes[1].set_title("Reconstructed image (10%)")
axes[1].axis("off")

plt.show()

# sjekker flere prosenter
percentages = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for ax, percentage in zip(axes.ravel(), percentages):

    coefficients_to_keep = int(total_coefficients * percentage)

    threshold = np.partition(
        magnitude.flatten(),
        -coefficients_to_keep
    )[-coefficients_to_keep]

    compression_mask = magnitude >= threshold
    compressed_fourier = shifted_fourier_image * compression_mask

    unshifted_compressed = np.fft.ifftshift(compressed_fourier)
    reconstructed_image = np.fft.ifft2(unshifted_compressed)
    reconstructed_image = np.real(reconstructed_image)

    ax.imshow(reconstructed_image, cmap="gray")
    ax.set_title(f"{percentage * 100:.1f}% coefficients")
    ax.axis("off")

plt.tight_layout()
plt.show()
