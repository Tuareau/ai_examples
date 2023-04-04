import random
import numpy as np
import matplotlib.pyplot as plt

def visualize_kmeans_clustering(X, max_iters=1000):

    # Задаем число кластеров
    k = 4

    # Инициализируем центры кластеров случайным образом
    centers = np.array(random.sample(list(X), k))

    # Функция для расчета расстояний между точками
    def dist(a, b):
        # Евклидово расстояние
        # return np.linalg.norm(a - b) 
        n = len(a)
        return sum((a[i] - b[i])**2 for i in range(n))**0.5

    # Основной цикл алгоритма K-means
    for i in range(max_iters):
        # Инициализируем список, содержащий пустые массивы для каждого кластера
        clusters = [[] for _ in range(k)]
    
        # Для каждой точки данных находим ближайший центр кластера
        for x in X:
            distances = [dist(x, center) for center in centers]
            closest_center_index = np.argmin(distances)
            clusters[closest_center_index].append(x)
                    
        # Пересчитываем центры кластеров
        new_centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        
        # Проверить, изменились ли центроиды, и если нет - выйти из цикла
        if np.allclose(centers, new_centers):
            break
        
        # Обновляем центры кластеров
        centers = new_centers 

    # Получаем метки кластеров для каждого объекта данных
    labels = np.zeros(len(X))
    for i in range(len(X)):
        distances = [dist(X[i], center) for center in centers]
        closest_center_index = np.argmin(distances)
        labels[i] = closest_center_index

    # Визуализируем результаты
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, linewidths=3, color='magenta')
    plt.title('K-means Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()