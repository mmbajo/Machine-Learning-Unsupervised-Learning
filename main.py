import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt('netflix_complete.txt')

# TODO: Your code here
K = 12
# mixture, post = common.init(X, K, 0)
# mixture, post, cost = em.run(X, mixture, post)
# print('seed : 0 ->', cost)

mixture, post = common.init(X, K, 1)
mixture, post, cost = em.run(X, mixture, post)
X_pred = em.fill_matrix(X, mixture)
rmse = common.rmse(X_gold, X_pred)
print('seed : 1 ->', rmse)

# mixture, post = common.init(X, K, 2)
# mixture, post, cost = em.run(X, mixture, post)
# print('seed : 2 ->', cost)

# mixture, post = common.init(X, K, 3)
# mixture, post, cost = em.run(X, mixture, post)
# print('seed : 3 ->', cost)

# mixture, post = common.init(X, K, 4)
# mixture, post, cost = em.run(X, mixture, post)
# print('seed : 4 ->', cost)