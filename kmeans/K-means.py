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


# ==========K-Means===================
_random = random.choices([i for i in range(len(X))], k=centers)
center_sets = X[_random, :]
center_label = y[_random]
# plot(selected_points, selected_label)

distance = lambda point, sets: [np.sum((i - point) ** 2) ** 0.5 for i in sets]


def Distance(point_sets: np.array, center_sets: np.array):
    for i in range(len(point_sets)):
        dist = np.array(
            sorted(
                enumerate(distance(point_sets[i, :], center_sets)),
                key=lambda x: (x[1], x[0]),
            )
        )
        yield dist[0]


import functools, itertools
from pprint import pprint
from copy import deepcopy

past_center = None
for _ in itertools.count():
    # best adjacent
    y = np.array(list(Distance(X, center_sets)))[:, 0]
    full_data = sorted(list(zip(X[:], y[:])), key=lambda x: x[-1])
    # updated X and label y
    X, y = zip(*full_data)
    X = np.array(X)
    y = np.array(y)
    # update new center_sets
    for label in range(centers):
        cluster = list(filter(lambda x: x[-1] == label, full_data))
        X_partial, y_partial = zip(*cluster)
        center_sets[label] = np.mean(X_partial, axis=0)
        center_label[label] = label

    if past_center is not None and np.array_equal(past_center, center_sets):
        break
    past_center = deepcopy(center_sets)
    plot(center_sets, center_label)
