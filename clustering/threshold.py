import random
import numpy as np
import matplotlib.pyplot as plt

# Функция для вычисления расстояния между объектами
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Функция для реализации пороговой кластеризации
def threshold_clustering(X, distance_threshold):
    # Создаем первый кластер, содержащий первый объект
    clusters = [[0]]
    # Для каждого объекта
    for i in range(1, len(X)):
        # Находим расстояние от текущего объекта до центра каждого кластера
        distances = [distance(X[i], X[c].mean(axis=0)) for c in clusters]
        # Если расстояние до ближайшего кластера меньше порогового значения,
        # добавляем объект в этот кластер
        if min(distances) <= distance_threshold:
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(i)
        # Иначе создаем новый кластер, содержащий текущий объект
        else:
            clusters.append([i])
    # Присваиваем метки кластеров
    labels = np.zeros(len(X))
    for i, c in enumerate(clusters):
        for j in c:
            labels[j] = i
    return labels

def visualize_threshold_clustering(X, distance_threshold):
    # Выполняем пороговую кластеризацию
    labels = threshold_clustering(X, distance_threshold)

    # Выводим результаты кластеризации
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.title('Threshold Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()