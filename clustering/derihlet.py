import numpy as np
import matplotlib.pyplot as plt

def visualize_dpmm_clustering(X):

    # Инициализация
    n_clusters = 5
    n_features = X.shape[1]
    n_samples = X.shape[0]
    alpha = 1
    mu = np.random.randn(n_clusters, n_features)
    cov = np.tile(np.identity(n_features), (n_clusters, 1, 1))
    r = np.ones((n_samples, n_clusters)) / n_clusters

    # E-шаг и M-шаг
    for i in range(50):
        # E-шаг
        for j in range(n_clusters):
            d = X - mu[j]
            cov_inv = np.linalg.inv(cov[j])
            cov_det = np.linalg.det(cov[j])
            norm_const = 1.0 / np.sqrt((2*np.pi)**n_features * cov_det)
            exponent = np.sum(-0.5 * np.dot(d, cov_inv) * d, axis=1)
            r[:, j] = norm_const * np.exp(exponent)
        r *= alpha
        r /= r.sum(axis=1)[:, np.newaxis]

        # M-шаг
        n = r.sum(axis=0)
        alpha = n_samples / (n_samples + alpha)
        for j in range(n_clusters):
            mu[j] = (r[:, j][:, np.newaxis] * X).sum(axis=0) / n[j]
            cov[j] = np.dot((r[:, j][:, np.newaxis] * (X - mu[j])).T, X - mu[j]) / n[j]

    # Проверка сходимости
    labels = r.argmax(axis=1)

    # Вывод результата на график
    plt.scatter(X[:, 0], X[:, 1], c=labels, alpha=1)
    plt.scatter(mu[:, 0], mu[:, 1], s=100, c='magenta', marker='x')
    plt.title("DPMM Clustering")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
