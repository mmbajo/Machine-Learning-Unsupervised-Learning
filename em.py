"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    p_x = np.zeros((n, K))
    cu_ones_matrix = X!=0
    dims = np.sum(cu_ones_matrix, axis = 1)
    
    for i in range(n):
        for k in range(K):
            filter = np.where(X[i, :] != 0)
            #p_x[i, k] = np.log(mixture.p[k] + 1e-16) + np.log((1/((2 * np.pi * mixture.var[k])**(dims[i]/2))) * np.exp(-((np.inner((X[i,:][filter]-mixture.mu[k,:][filter]), (X[i,:][filter]-mixture.mu[k,:][filter])))/(2*mixture.var[k]))))
            p_x[i, k] = np.log(mixture.p[k] + 1e-16) + (dims[i]/2)* np.log((1/((2 * np.pi * mixture.var[k])))) - ((np.inner((X[i,:][filter]-mixture.mu[k,:][filter]), (X[i,:][filter]-mixture.mu[k,:][filter])))/(2*mixture.var[k]))
    #print(p_x)
    minus_log_summation = logsumexp(p_x, axis = 1, keepdims = True)
    post = np.exp(p_x - minus_log_summation)
    #LL = np.sum(np.log(np.sum(np.exp(p_x), axis = 1)))
    LL = np.sum(minus_log_summation)
    return post, LL



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    indicator = X!=0

    mu = mixture.mu

    for k in range(K):
        for col in range(d):
            if np.dot(post[:, k], indicator[:, col]) <= 1:
                mu[k, col] = mu[k, col]
                #mu[k, col] = np.dot(np.multiply(post[:, k], indicator[:, col]), X[:, col])/np.dot(post[:, k], indicator[:, col])

            else:
                mu[k, col] = np.dot(np.multiply(post[:, k], indicator[:, col]), X[:, col])/np.dot(post[:, k], indicator[:, col])

    ## var caluculation
    normalizer = np.sum(np.multiply(post, np.sum(indicator, axis = 1, keepdims = True)), axis = 0)
    temp = np.zeros((n, K))

    for i in range(n):
        for k in range(K):
            filter = np.where(X[i, :] != 0)
            temp[i, k] = np.dot((X[i, :][filter] - mu[k, :][filter]), (X[i, :][filter] - mu[k, :][filter])) * post[i, k]

    summation = temp.sum(axis = 0)
    var = np.divide(summation, normalizer)
    var = np.maximum(var, min_variance)
    n_k = np.sum(post, axis = 0)
    p = n_k / n
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_cost = None
    new_cost = None
    while (old_cost is None or ((new_cost - old_cost)/abs(new_cost) >= 10**(-6))):
        old_cost = new_cost
        post, new_cost = estep(X, mixture)
        mixture = mstep(X, post, mixture)
    return mixture, post, new_cost
    #raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    post, _ = estep(X, mixture)
    X_indicator = X != 0

    post_sum_by_k = np.sum(post, axis = 1)

    X_predict = np.zeros((n, d))

    for row in range(n):
        for col in range(d):
            if X_indicator[row, col] == 1:
                X_predict[row, col] = X[row, col]
            else:
                X_predict[row, col] = np.dot(post[row, :], mixture.mu[:, col])/post_sum_by_k[row]
    return X_predict

