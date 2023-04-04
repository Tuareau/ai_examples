import kmeans
import threshold
import derihlet

import numpy as np

# Генерируем случайные данные
np.random.seed(0)
X = np.random.randn(1000, 2) * 10

# Визуализация K-means
kmeans.visualize_kmeans_clustering(X);

# Визуализация Threshold
threshold.visualize_threshold_clustering(X, distance_threshold=15);

# Визуализация DPMM
derihlet.visualize_dpmm_clustering(X)
