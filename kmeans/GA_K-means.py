import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 遗传算法参数
POPULATION_SIZE = 50  # 种群大小
NUM_GENERATIONS = 100  # 迭代次数
MUTATION_RATE = 0.4  # 突变概率
K = 4  # 聚类的数量


def initialize_population(data, k):
    population = []
    for _ in range(POPULATION_SIZE):
        individual = data[np.random.choice(data.shape[0], k, replace=False)]
        population.append(individual)
    return np.array(population)


def fitness(individual, data):
    kmeans = KMeans(n_clusters=len(individual), init=individual, n_init=1, max_iter=100)
    kmeans.fit(data)
    return -kmeans.inertia_


def select_parents(population, fitness_scores):
    fitness_sum = np.sum(fitness_scores)
    selection_probs = fitness_scores / fitness_sum
    parent_indices = np.random.choice(len(population), size=2, p=selection_probs)
    return population[parent_indices]


def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


def mutate(individual, data):
    if np.random.rand() < MUTATION_RATE:
        mutate_index = np.random.randint(len(individual))
        individual[mutate_index] = data[np.random.choice(data.shape[0])]
    return individual


def genetic_algorithm_kmeans(data, k):
    population = initialize_population(data, k)

    for generation in range(NUM_GENERATIONS):
        fitness_scores = np.array(
            [fitness(individual, data) for individual in population]
        )
        new_population = []

        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, data)
            child2 = mutate(child2, data)
            new_population.append(child1)
            new_population.append(child2)

        population = np.array(new_population)
        best_fitness = np.max(fitness_scores)
        print(f"Generation {generation+1}: Best Fitness = {-best_fitness}")

    best_individual_index = np.argmax(fitness_scores)
    best_individual = population[best_individual_index]
    final_kmeans = KMeans(n_clusters=k, init=best_individual, n_init=1, max_iter=300)
    final_kmeans.fit(data)

    return final_kmeans


def plot_clusters(data, kmeans):
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap="viridis")
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=300,
        c="red",
        marker="X",
    )
    plt.title("K-Means Clustering with GA Optimization")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


# 示例数据
from sklearn.datasets import make_blobs
import random

n_samples = 100
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

data = X

# 运行遗传算法优化的K-Means
kmeans_model = genetic_algorithm_kmeans(data, K)

# 绘制聚类结果
plot_clusters(data, kmeans_model)
