
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random

n_samples = 300
n_features = 2
centers = 4
cluster_std = 1.0

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
"""kmeans/K-means-enhancement.py for the adjacent cluster center,
"""

def get_most_adjacent_pair(center_sets: np.array):
    # non-DC implement, low K and efficient
    min_distance = float('inf')
    id1, id2 = -1, -1
    num_points = len(center_sets)
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(center_sets[i] - center_sets[j])
            if distance < min_distance:
                min_distance = distance
                id1, id2 = i, j

    return id1, id2
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

    # if past_center is not None and np.array_equal(past_center, center_sets):
    #     break
    # past_center = deepcopy(center_sets)
    plot(center_sets, center_label)
    most_adjacent_cluster_pair= get_most_adjacent_pair(center_sets)
    ids= most_adjacent_cluster_pair[0]
    ## select one from other cluster randomly
    other_cluster = list(filter(lambda x: x[-1] !=ids , full_data))
    pprint(random.choice(other_cluster))
    choice_X,choice_y = random.choice(other_cluster)
    center_sets[ids]=choice_X
    center_label[ids]=choice_y
    




