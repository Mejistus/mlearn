import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random

n_samples = 100
n_features = 2
centers = 4
cluster_std = 3.0

X, y = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=centers,
    cluster_std=cluster_std,
    random_state=42,
)


def plot(center_points, center_label):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="autumn", s=5)
    if center_points and center_points:
        plt.scatter(
            center_points[:, 0],
            center_points[:, 1],
            c=center_label,
            cmap="autumn",
            s=200,
            marker="*",
        )
    plt.title("Simple 2D Dataset for K-means Testing")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


plot([], [])
