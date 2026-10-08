import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color, transform

folder = "vehicle_type_recognition/motorcycle"

# -------------------------
# 1. Load and preprocess images
# -------------------------

image_files = os.listdir(folder)

images = []
# Gjør alle bildene i mappen om til gray scale og gjør alle bildene om til 64x64 piksler
for file in image_files:

    path = os.path.join(folder, file)

    image = io.imread(path)

    gray_image = color.rgb2gray(image)

    gray_image = transform.resize(gray_image, (64, 64))

    images.append(gray_image)

# -------------------------
# 2. Prepare data for PCA
# -------------------------

image_matrix = []
# Gjør alle bildene om til en matrise der hver rad er et bilde, og hver kolonne er en pikselverdi
for image in images:
    flattened_image = image.flatten()
    image_matrix.append(flattened_image)

image_matrix = np.array(image_matrix)

mean_image = np.mean(image_matrix, axis=0)
centered_data = image_matrix - mean_image

covariance_matrix = np.cov(centered_data, rowvar=False)

# -------------------------
# 3. PCA implementation
# -------------------------

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

# -------------------------
# 4. Reconstruction
# -------------------------

def reconstruct_pca(k):
    components = eigenvectors[:, :k]

    projected = centered_data @ components

    reconstructed = projected @ components.T
    reconstructed = reconstructed + mean_image

    return reconstructed

k = 20

# Rekonstruer bildene med 20 principal components
reconstructed_data = reconstruct_pca(k)

# print("Reconstructed data shape:", reconstructed_data.shape)

# Gjør første rekonstruerte bilde tilbake til 64 x 64
reconstructed_image = reconstructed_data[0].reshape(64, 64)

# Velg tre eksempelbilder fra datasettet
indices = [0, 20, 50]

plt.figure(figsize=(8, 9))

for i, index in enumerate(indices):
    # Originalt preprosessert bilde
    plt.subplot(3, 2, 2*i + 1)
    plt.imshow(images[index], cmap="gray")
    plt.title("PCA input")
    plt.axis("off")

    # Rekonstruert bilde
    reconstructed_image = reconstructed_data[index].reshape(64, 64)

    plt.subplot(3, 2, 2*i + 2)
    plt.imshow(reconstructed_image, cmap="gray")
    plt.title(f"Reconstructed, k={k}")
    plt.axis("off")

plt.tight_layout()
# plt.show()

# -------------------------
# 5. Experimentation
# -------------------------

k_values = [5, 10, 20, 50, 100]
image_index = 0

plt.figure(figsize=(15, 4))

# Vis PCA-input
plt.subplot(1, len(k_values) + 1, 1)
plt.imshow(images[image_index], cmap="gray")
plt.title("PCA input")
plt.axis("off")

# Vis rekonstruksjon for ulike k-verdier
for i, k in enumerate(k_values):
    reconstructed = reconstruct_pca(k)
    reconstructed_image = reconstructed[image_index].reshape(64, 64)

    plt.subplot(1, len(k_values) + 1, i + 2)
    plt.imshow(reconstructed_image, cmap="gray")
    plt.title(f"k = {k}")
    plt.axis("off")

plt.tight_layout()
# plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance
)

plt.xlim(0, 150)

plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative explained variance")

plt.grid()
# plt.show()

for threshold in [0.80, 0.90, 0.95]:
    k_needed = np.argmax(cumulative_variance >= threshold) + 1
    print(f"{threshold * 100:.0f}% variance: {k_needed} components")


# -------------------------
# 6. Visual Analysis
# -------------------------

indices = [0, 20, 50]
k_values = [5, 20, 63, 100]

plt.figure(figsize=(12, 9))

for row, index in enumerate(indices):

    # PCA input
    plt.subplot(len(indices), len(k_values) + 1,
                row * (len(k_values) + 1) + 1)
    plt.imshow(images[index], cmap="gray")
    plt.title("PCA input")
    plt.axis("off")

    # Reconstructions
    for col, k in enumerate(k_values):
        reconstructed = reconstruct_pca(k)
        reconstructed_image = reconstructed[index].reshape(64, 64)

        plt.subplot(
            len(indices),
            len(k_values) + 1,
            row * (len(k_values) + 1) + col + 2
        )

        plt.imshow(reconstructed_image, cmap="gray")
        plt.title(f"k={k}")
        plt.axis("off")

plt.tight_layout()
# plt.show()


# -------------------------
# 6. Error Analysis
# -------------------------

k_values = [5, 20, 38, 63, 84, 100]

for k in k_values:
    reconstructed = reconstruct_pca(k)
    mse = np.mean((image_matrix - reconstructed) ** 2)

    print(f"k={k}: MSE={mse:.6f}")

k_values = [5, 20, 38, 63, 84, 100]
mse_values = []

for k in k_values:
    reconstructed = reconstruct_pca(k)
    mse = np.mean((image_matrix - reconstructed) ** 2)
    mse_values.append(mse)

plt.figure(figsize=(8, 5))
plt.plot(k_values, mse_values, marker="o")

plt.xlabel("Number of principal components")
plt.ylabel("Mean Squared Error")
plt.title("Reconstruction error for different values of k")
plt.grid()

plt.show()