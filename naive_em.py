"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape

    p_x = np.zeros((n, K))

    for i in range(n):
        for k in range(K):
            p_x[i, k] = mixture.p[k] * (1/((2 * np.pi * mixture.var[k])**(d/2))) * np.exp(-((np.inner((X[i,:]-mixture.mu[k,:]), (X[i,:]-mixture.mu[k,:])))/(2*mixture.var[k])))

    p_theta = p_x.sum(axis = 1)
    posterior = np.divide(p_x, p_theta.reshape(-1,1))
    loglikelihood = np.sum(posterior * np.log(np.divide(p_x, posterior)))
    return posterior, loglikelihood
    #raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    n_k = np.sum(post, axis = 0)
    mu_k = np.divide(np.dot(post.T, X), n_k.reshape(-1, 1))
    
    temp = np.zeros((n, K))
    
    for i in range(n):
        for k in range(K):
            temp[i, k] = np.dot((X[i, :] - mu_k[k, :]), (X[i, :] - mu_k[k, :])) * post[i, k]
    summation = temp.sum(axis = 0)
    #summation = np.hstack([np.sum(np.dot(np.inner((X[i, :] - mu_k), (X[i, :] - mu_k)), post[i, :])) for i in range(n)])
    #print(summation)
    var_k = np.divide(summation, n_k)/d
    p_k = n_k / n
    #print(n_k)
    return GaussianMixture(mu_k, var_k, p_k)


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
        mixture = mstep(X, post)
    return mixture, post, new_cost
    #raise NotImplementedError
