import random
import numpy as np
import matplotlib.pyplot as plt

# ������� ��� ���������� ���������� ����� ���������
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# ������� ��� ���������� ��������� �������������
def threshold_clustering(X, distance_threshold):
    # ������� ������ �������, ���������� ������ ������
    clusters = [[0]]
    # ��� ������� �������
    for i in range(1, len(X)):
        # ������� ���������� �� �������� ������� �� ������ ������� ��������
        distances = [distance(X[i], X[c].mean(axis=0)) for c in clusters]
        # ���� ���������� �� ���������� �������� ������ ���������� ��������,
        # ��������� ������ � ���� �������
        if min(distances) <= distance_threshold:
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(i)
        # ����� ������� ����� �������, ���������� ������� ������
        else:
            clusters.append([i])
    # ����������� ����� ���������
    labels = np.zeros(len(X))
    for i, c in enumerate(clusters):
        for j in c:
            labels[j] = i
    return labels

def visualize_threshold_clustering(X, distance_threshold):
    # ��������� ��������� �������������
    labels = threshold_clustering(X, distance_threshold)

    # ������� ���������� �������������
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.title('Threshold Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()