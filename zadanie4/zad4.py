from typing import List
import numpy as np
from collections import Counter

class Solution:

    # 1. K-Nearest Neighbors KNN
    def knn_class(self, X_train, y_train, k, X_test):
        y_test = []
        for q in X_test:
            distances = []
            for x in X_train:
                distances.append(np.linalg.norm(x - q))
            k_indices = np.argsort(distances)[:k]
            k_labels = [y_train[i] for i in k_indices]
            y_test.append(sorted(Counter(k_labels).items(), reverse=True, key=lambda l: l[1])[0][0])
        return y_test

    # 2. KNN for centroids
    def knn_cluster(self, X, k):
        centroids = np.mean(X, axis=0)
        distances = [np.linalg.norm(x - centroids) for x in X]
        indices = np.argsort(distances)[:k]
        labels = np.zeros(len(X), dtype=int)
        labels[indices] = 1
        return labels

    # 3. Principal Component Analysis PCA
    def pca(self, X, n_components):

        if len(X) < n_components:
            raise ValueError
            
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        
        selected_eigenvectors = sorted_eigenvectors[:, :n_components]
        X_reduced = np.dot(X_centered, selected_eigenvectors)
        
        explained_variance = sorted_eigenvalues[:n_components] / np.sum(eigenvalues)
        
        return X_reduced, explained_variance


np.random.seed(0)
X_train = np.random.rand(100, 7)
y_train = np.random.randint(0, 5, 100)
X_test = np.random.rand(5, 7)

sol = Solution()
k = 3
y_test_predicted = sol.knn_class(X_train, y_train, k, X_test)
print("Wynik klastrowania:", y_test_predicted)
print()


# Testing KNN for centroids
labels = sol.knn_cluster(X_train, k)
print("Wynik klastrowania:", labels)
print()

# Testing PCA
X = np.random.randn(100, 10) 
X_reduced, variance_explained = sol.pca(X, n_components=2)
print("Zredukowane dane:", X_reduced)
print("Wyjaśniona wariancja:", variance_explained)