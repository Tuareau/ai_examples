import random
import numpy as np
import matplotlib.pyplot as plt

def visualize_kmeans_clustering(X, max_iters=1000):

    # ������ ����� ���������
    k = 4

    # �������������� ������ ��������� ��������� �������
    centers = np.array(random.sample(list(X), k))

    # ������� ��� ������� ���������� ����� �������
    def dist(a, b):
        # ��������� ����������
        # return np.linalg.norm(a - b) 
        n = len(a)
        return sum((a[i] - b[i])**2 for i in range(n))**0.5

    # �������� ���� ��������� K-means
    for i in range(max_iters):
        # �������������� ������, ���������� ������ ������� ��� ������� ��������
        clusters = [[] for _ in range(k)]
    
        # ��� ������ ����� ������ ������� ��������� ����� ��������
        for x in X:
            distances = [dist(x, center) for center in centers]
            closest_center_index = np.argmin(distances)
            clusters[closest_center_index].append(x)
                    
        # ������������� ������ ���������
        new_centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        
        # ���������, ���������� �� ���������, � ���� ��� - ����� �� �����
        if np.allclose(centers, new_centers):
            break
        
        # ��������� ������ ���������
        centers = new_centers 

    # �������� ����� ��������� ��� ������� ������� ������
    labels = np.zeros(len(X))
    for i in range(len(X)):
        distances = [dist(X[i], center) for center in centers]
        closest_center_index = np.argmin(distances)
        labels[i] = closest_center_index

    # ������������� ����������
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, linewidths=3, color='magenta')
    plt.title('K-means Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()