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

mask[distance_from_center <= radius] = 1 # hvis avstanden fra sentrum er mindre enn radius, sett verdien til 1

# viser masken 
plt.imshow(mask, cmap="gray")
plt.show()

filtered_fourier = shifted_fourier_image * mask # bruker masken til å filtrere frekvensbildet
unshifted_fourier = np.fft.ifftshift(filtered_fourier)
filtered_image = np.fft.ifft2(unshifted_fourier)

filtered_image = np.abs(filtered_image)

# viser low-pass filtrert bilde
plt.imshow(filtered_image, cmap="gray")
plt.title("Low-pass filtered image")
plt.axis("off")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(gray_image, cmap="gray")
axes[0].set_title("Original image")
axes[0].axis("off")

axes[1].imshow(filtered_image, cmap="gray")
axes[1].set_title("Low-pass filtered image")
axes[1].axis("off")

plt.show()