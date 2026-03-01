import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Given parameters
mu1 = np.array([2, 3])
sigma1 = np.array([[2, 0.5], [0.5, 1]])

mu2 = np.array([4, 4])
sigma2 = np.array([[1.5, -0.3], [-0.3, 0.8]])

priors = [0.1, 0.25, 0.5, 0.75, 0.9]

# Generate grid for contour plot
x, y = np.meshgrid(np.linspace(0, 6, 500), np.linspace(0, 6, 500))
pos = np.dstack((x, y))

# Create contour plots for each class
rv1 = multivariate_normal(mu1, sigma1)
rv2 = multivariate_normal(mu2, sigma2)

plt.figure(figsize=(10, 6))

# Contour plots
contour1 = plt.contour(x, y, rv1.pdf(pos), levels=10, cmap='Blues', alpha=0.6)
contour2 = plt.contour(x, y, rv2.pdf(pos), levels=10, cmap='Reds', alpha=0.6)

# Adding legends manually
handles = [
    plt.Line2D([0, 1], [0, 1], color='blue', alpha=0.6, label='Class 1'),
    plt.Line2D([0, 1], [0, 1], color='red', alpha=0.6, label='Class 2')
]
plt.legend(handles=handles)
plt.title('Contour plots for each class')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Plot with decision boundaries for different priors
plt.figure(figsize=(10, 6))

# Contour plots
plt.contour(x, y, rv1.pdf(pos), levels=10, cmap='Blues', alpha=0.3)
plt.contour(x, y, rv2.pdf(pos), levels=10, cmap='Reds', alpha=0.3)

# Decision boundaries
colors = ['green', 'purple', 'orange', 'c', 'black']
for i, prior in enumerate(priors):
    decision_boundary = np.log(rv1.pdf(pos) * prior / (rv2.pdf(pos) * (1 - prior)))
    plt.contour(x, y, decision_boundary, levels=[0], colors=colors[i], alpha=0.8)

# Adding legends manually
legend_handles = [
    plt.Line2D([0, 1], [0, 1], color=colors[i], alpha=0.8, label=f'P(ω1)={prior}')
    for i, prior in enumerate(priors)
]
plt.legend(handles=handles + legend_handles)
plt.title('Decision boundaries for different priors')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Same covariance matrix for both classes
sigma_common = np.array([[1.2, 0.4], [0.4, 1.2]])

rv1_common = multivariate_normal(mu1, sigma_common)
rv2_common = multivariate_normal(mu2, sigma_common)

plt.figure(figsize=(10, 6))

# Contour plots
contour1_common = plt.contour(x, y, rv1_common.pdf(pos), levels=10, cmap='Blues', alpha=0.6)
contour2_common = plt.contour(x, y, rv2_common.pdf(pos), levels=10, cmap='Reds', alpha=0.6)

# Decision boundaries
for i, prior in enumerate(priors):
    decision_boundary_common = np.log(rv1_common.pdf(pos) * prior / (rv2_common.pdf(pos) * (1 - prior)))
    plt.contour(x, y, decision_boundary_common, levels=[0], colors=colors[i], alpha=0.8)

# Adding legends manually
plt.legend(handles=handles + legend_handles)
plt.title('Contours and decision boundaries with same covariance matrix')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
