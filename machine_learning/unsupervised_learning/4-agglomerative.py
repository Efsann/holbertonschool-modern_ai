#!/usr/bin/env python3
"""
Agglomerative Hierarchical Clustering Module
"""
from sklearn import cluster
from sklearn import metrics
Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(X, n_clusters, random_state, n_components,
                             use_pca_data=True):
    """
    Performs Agglomerative hierarchical clustering on tabular data
    Arguments:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
        n_clusters (int): Number of clusters
        random_state (int): Random seed for reproducibility (not directly
                            used by AgglomerativeClustering, but kept for API)
        n_components (int): Number of PCA components to reduce the data to
        use_pca_data (bool): Whether to apply PCA to reduce dimensionality
    Returns:
        sklearn.cluster.AgglomerativeClustering: Fitted clustering instance
        numpy.ndarray: Data used for fitting (PCA-reduced or original)
        float: Silhouette score of the clustering (None if n_clusters=1)
    """
    if use_pca_data:
        X_used, _ = Apply_PCA(X, n_components=n_components,
                              random_state=random_state)
    else:
        X_used = X

    agg_cluster = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                                  linkage='ward')
    agg_cluster.fit(X_used)

    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, agg_cluster.labels_)
    else:
        score = None

    return agg_cluster, X_used, score
