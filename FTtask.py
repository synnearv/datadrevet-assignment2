from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np

# Oppgave 1
image = io.imread("vehicle_type_recognition/hatchback/PIC_1.jpg")
gray_image = color.rgb2gray(image)

# viser det originale gråtonede bildet
plt.imshow(gray_image, cmap='gray')
plt.show()
print(image.shape)
print(gray_image.shape)
print(gray_image)

fourier_image = np.fft.fft2(gray_image)
shifted_fourier_image = np.fft.fftshift(fourier_image)
magnitude = np.abs(shifted_fourier_image)
magnitude_spectrum = np.log1p(magnitude)

print(fourier_image)
print(shifted_fourier_image)
print(magnitude_spectrum)

# viser spektrumet av frekvensbildet
plt.imshow(magnitude_spectrum, cmap='gray')
plt.show()


