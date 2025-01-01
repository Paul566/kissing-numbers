import numpy as np
from scipy.spatial import KDTree


def func(X):
    squared_norms = np.sum(X**2, axis=1)
    squared_dists = squared_norms[:, np.newaxis] + squared_norms - 2 * X @ X.T
    squared_dists = np.minimum(squared_dists, 1)
    potentials = (1 - squared_dists)**2
    return (np.sum(potentials) - X.shape[0]) / 2

def grad(X):
    squared_norms = np.sum(X**2, axis=1)
    squared_dists = squared_norms[:, np.newaxis] + squared_norms - 2 * X @ X.T
    squared_dists = np.minimum(squared_dists, 1)
    multipliers = 4 * (squared_dists - 1)
    
    differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    grads = differences * multipliers[:, :, np.newaxis]
    return np.sum(grads, axis=1)

def renorm(X):
    # makes X[i] unit norm
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / row_norms

def gd(grad, x0, num_iter, tol, alpha):
    x = x0.copy()
    for iter_idx in range(num_iter):
        g = grad(x)
        next_x = renorm(x - alpha * g)
        if np.linalg.norm((next_x - x) / alpha) < tol:
            break
        x = next_x
    return x

def grad_kdtree(X, inds):
    differences = X[:, np.newaxis, :] - X[inds[:, 1:]]
    
    dists = np.minimum(np.linalg.norm(differences, axis=2), 1)
    multipliers = 4 * (dists**2 - 1)
    
    grads = differences * multipliers[:, :, np.newaxis]
    return np.sum(grads, axis=1)

def heavy_ball(grad, x0, num_iter, tol, alpha, beta):
    x = x0.copy()
    for iter_idx in range(num_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        if len(conv) > 1:
            next_x = renorm(x - alpha * g + beta * (x - conv[-2]))
        else:
            next_x = renorm(x - alpha * g)

        if np.linalg.norm((next_x - x) / alpha) < tol:
            break
        x = next_x
    return x

def accelerated_gradient(func, grad, x0, num_iter, tol, alpha, restart=False, adapt_step=False, alpha_min=1e-3):
    x = x0.copy()
    y = x0.copy()
    k = 0
    for iter_idx in range(num_iter):
        x_next = y - alpha * grad(y)
        grad_x = grad(x)
        if restart and func(x) < func(renorm(x_next)):
            if adapt_step and alpha > alpha_min:
                if func(x) < func(renorm(x - alpha * grad_x)):
                    alpha = alpha * 0.8
            x_next = x - alpha * grad_x
            k = 0
        y = x_next + (k + 1) / (k + 4) * (x_next - x)
        y = renorm(y)
        x_next = renorm(x_next)
        if np.linalg.norm((x - x_next) / alpha) < tol:
            break
        x = x_next.copy()
        g = grad(x)
        k += 1
    return x

def accelerated_gradient_kdt(func, x0, recomp_tree, num_neighbors, num_iter, tol, alpha, restart=False, adapt_step=False, alpha_min=1e-3):
    x = x0.copy()
    y = x0.copy()
    k = 0

    tree = KDTree(x)
    dists, inds = tree.query(x, k=num_neighbors)
    
    for iter_idx in range(num_iter):
        if iter_idx % recomp_tree == 0:
            tree = KDTree(x)
            dists, inds = tree.query(x, k=num_neighbors)
        
        x_next = y - alpha * grad_kdtree(y, inds)
        grad_x = grad_kdtree(x, inds)
        if restart and func(x) < func(renorm(x_next)):
            if adapt_step and alpha > alpha_min:
                if func(x) < func(renorm(x - alpha * grad_x)):
                    alpha = alpha * 0.8
            x_next = x - alpha * grad_x
            k = 0
        y = x_next + (k + 1) / (k + 4) * (x_next - x)
        y = renorm(y)
        x_next = renorm(x_next)
        if np.linalg.norm((x - x_next) / alpha) < tol:
            break
        x = x_next.copy()
        g = grad_kdtree(x, inds)
        k += 1
    return x
    