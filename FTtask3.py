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

# Oppgave 3: high-pass filter
# fyller matrisen med 1, og setter verdien til 0 for de pikslene som er innenfor radiusen
high_pass_mask = np.ones((rows, cols))
high_pass_mask[distance_from_center <= radius] = 0

# viser high-pass filter masken
plt.imshow(high_pass_mask, cmap="gray")
plt.title("High-pass filter mask")
plt.show()

high_pass_fourier = shifted_fourier_image * high_pass_mask # bruker high-pass masken til å filtrere frekvensbildet
unshifted_high_pass = np.fft.ifftshift(high_pass_fourier) # flytter frekvensbildet tilbake til opprinnelig posisjon

high_pass_image = np.fft.ifft2(unshifted_high_pass) # beregner invers Fourier-transformasjon for å få tilbake bildet i romdomenet
high_pass_image = np.real(high_pass_image) # tar den reelle delen av det komplekse resultatet

# Viser bildet
plt.imshow(high_pass_image, cmap="gray")
plt.title("High-pass filtered image")
plt.axis("off")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(gray_image, cmap="gray")
axes[0].set_title("Original image")
axes[0].axis("off")

axes[1].imshow(high_pass_image, cmap="gray")
axes[1].set_title("High-pass filtered image")
axes[1].axis("off")

plt.show()