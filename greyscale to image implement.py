# Step 1: Upload image and install packages
from google.colab import files
uploaded = files.upload()

!pip install opencv-python-headless matplotlib numpy

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (auto-detect uploaded filename)
filename = list(uploaded.keys())[0]
color_img = cv2.imread(filename)
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Resize image to speed up optimization (optional)
color_img = cv2.resize(color_img, (100, 100))

# Normalize color channels
R = color_img[:, :, 0].flatten() / 255.0
G = color_img[:, :, 1].flatten() / 255.0
B = color_img[:, :, 2].flatten() / 255.0

# Target grayscale (OpenCV standard)
target_gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY) / 255.0
target_gray_flat = target_gray.flatten()

# Fitness function: MSE between weighted grayscale and target grayscale
def fitness(weights):
    w_R, w_G, w_B = weights
    total = w_R + w_G + w_B
    if total == 0:
        return np.inf
    w_R /= total
    w_G /= total
    w_B /= total
    gray = w_R * R + w_G * G + w_B * B
    mse = np.mean((gray - target_gray_flat) ** 2)
    return mse

# Grey Wolf Optimizer (GWO) implementation
class GWO:
    def __init__(self, fitness_func, dim, n_wolves=20, max_iter=100):
        self.fitness_func = fitness_func
        self.dim = dim
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.positions = np.random.rand(n_wolves, dim)
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = np.inf
        self.beta_pos = np.zeros(dim)
        self.beta_score = np.inf
        self.delta_pos = np.zeros(dim)
        self.delta_score = np.inf

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.n_wolves):
                fitness = self.fitness_func(self.positions[i])
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()

            a = 2 - t * (2 / self.max_iter)
            for i in range(self.n_wolves):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    self.positions[i, j] = (X1 + X2 + X3) / 3

                self.positions[i] = np.clip(self.positions[i], 0, 1)

            if t % 10 == 0 or t == self.max_iter - 1:
                print(f"Iteration {t+1}/{self.max_iter}, Best MSE: {self.alpha_score:.8f}")

        return self.alpha_pos, self.alpha_score

# Run optimization
gwo = GWO(fitness_func=fitness, dim=3, n_wolves=30, max_iter=100)
best_weights, best_mse = gwo.optimize()

# Normalize weights
best_weights /= np.sum(best_weights)
print("\nOptimized Grayscale Weights:")
print(f"w_R = {best_weights[0]:.4f}, w_G = {best_weights[1]:.4f}, w_B = {best_weights[2]:.4f}")
print(f"Best MSE = {best_mse:.8f}")

# Generate grayscale image with optimized weights
optimized_gray = (best_weights[0] * color_img[:, :, 0] +
                  best_weights[1] * color_img[:, :, 1] +
                  best_weights[2] * color_img[:, :, 2]) / 255.0

# Plot images
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(color_img)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Target Grayscale (OpenCV)")
plt.imshow(target_gray, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Optimized Grayscale (GWO)")
plt.imshow(optimized_gray, cmap='gray')
plt.axis('off')

plt.show()
