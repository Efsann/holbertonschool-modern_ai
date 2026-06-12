#!/usr/bin/env python3
"""
Feature Standardization Module
"""
from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes tabular data using Scikit-learn
    Arguments:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
    Returns:
        numpy.ndarray: The standardized version of the input data
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
