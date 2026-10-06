import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color, transform

folder = "vehicle_type_recognition/motorcycle"

image_files = os.listdir(folder)

images = []
# Gjør alle bildene i mappen om til gray scale og gjør alle bildene om til 64x64 piksler
for file in image_files:

    path = os.path.join(folder, file)

    image = io.imread(path)

    gray_image = color.rgb2gray(image)

    gray_image = transform.resize(gray_image, (64, 64))

    images.append(gray_image)


image_matrix = []
# Gjør alle bildene om til en matrise der hver rad er et bilde, og hver kolonne er en pikselverdi
for image in images:
    flattened_image = image.flatten()
    image_matrix.append(flattened_image)

image_matrix = np.array(image_matrix)

mean_image = np.mean(image_matrix, axis=0)
centered_data = image_matrix - mean_image

covariance_matrix = np.cov(centered_data, rowvar=False)

# Finn egenverdier og egenvektorer fra kovariansmatrisen
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Finn rekkefølgen som sorterer egenverdiene fra størst til minst
sorted_indices = np.argsort(eigenvalues)[::-1]

# Sorter både egenverdiene og egenvektorene i samme rekkefølge
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# print("De 10 største egenverdiene:")
# print(eigenvalues[:10])

# Beregn forklart varians for hver egenverdi
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

cumulative_variance = np.cumsum(explained_variance_ratio)

# print("Kumulativ forklart varians:")
# print(cumulative_variance[:10])


k = 20
# Velg de k viktigste principal components
principal_components = eigenvectors[:, :k]

# Projiser bildene ned på de valgte principal components
# projected_data = centered_data @ principal_components

# print("Projected data shape:", projected_data.shape)

# Gjør PCA-representasjonen om tilbake til pikselrommet
# reconstructed_data = projected_data @ principal_components.T

# print("Reconstructed data shape:", reconstructed_data.shape)

