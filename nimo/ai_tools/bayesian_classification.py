"""Bayesian classification models, independent of acquisition functions.

RBF multiclass logistic regression with a Gaussian prior and a Laplace
posterior. Features use deterministic candidate-based centers in arbitrary dimensions.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
from scipy.linalg import cholesky, solve_triangular


def positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')


class RBFClassifier:
    def __init__(self, lengthscale=0.25, prior_sd=2.0,
                 num_centers=21, scaling=None):
        positive(lengthscale, 'lengthscale')
        positive(prior_sd, 'prior_sd')
        if type(num_centers) is not int or num_centers < 1:
            raise ValueError('num_centers must be a positive integer')
        scaling = 'minmax' if scaling is None else scaling
        if scaling not in ('minmax', 'none'):
            raise ValueError('scaling must be minmax or none')
        self.model, self.lengthscale, self.prior_sd = 'rbf', lengthscale, prior_sd
        self.num_centers, self.scaling = num_centers, scaling

    def features(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError('Features must be a nonempty finite numeric matrix')
        minimum = x.min(0) if self.scaling == 'minmax' else np.zeros(x.shape[1])
        span = x.max(0) - minimum if self.scaling == 'minmax' else np.ones(x.shape[1])
        span = np.where(span == 0, 1., span)
        transformed = (x - minimum) / span
        if not np.isfinite(transformed).all():
            raise ValueError('Feature scaling overflowed; rescale the input values')
        unique = np.unique(transformed, axis=0)  # Lexicographic order, independent of CSV ordering.
        chosen = [0]
        nearest = np.sum((unique - unique[0])**2, axis=1)
        for _ in range(1, min(self.num_centers, len(unique))):
            nearest[chosen] = -np.inf
            best = nearest.max()
            winner = int(np.flatnonzero(nearest >= best - 1e-12)[0])
            chosen.append(winner)
            nearest = np.minimum(nearest, np.sum((unique - unique[winner])**2, axis=1))
        centers = unique[chosen]
        rbf = np.exp(-np.sum((transformed[:, None, :] - centers[None, :, :])**2, axis=-1)
                     / (2*self.lengthscale**2))
        self.feature_metadata = dict(model=self.model, scaling=self.scaling,
            minimum=minimum.tolist(), span=span.tolist(), centers=centers.tolist(),
            lengthscale=self.lengthscale, prior_sd=self.prior_sd)
        return np.column_stack([np.ones(len(x)), rbf])

    def posterior_predictions(self, x, y, samples=4096, seed=0):
        """Return [posterior draw, candidate, class] probabilities.

        y uses -1 for unobserved candidates and contiguous 0..K-1 class IDs
        for observed labels. No unobserved labels are accepted by this API.
        """
        if type(samples) is not int or samples < 2:
            raise ValueError('samples must be an integer >= 2')
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(x) or not np.issubdtype(y.dtype, np.integer) or (y < -1).any():
            raise ValueError('y must contain integer class IDs or -1, with one entry per candidate')
        observed = np.flatnonzero(y >= 0)
        classes = np.unique(y[observed])
        if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError('At least two observed classes with contiguous IDs are required')
        phi = self.features(x)
        train, target, k = phi[observed], y[observed], len(classes)
        d = phi.shape[1] * k
        result = minimize(_objective, np.zeros(d), args=(train, target, k, self.prior_sd), jac=True,
                          method='L-BFGS-B', options={'maxiter':1000, 'ftol':1e-12, 'gtol':1e-7})
        if not result.success:
            raise RuntimeError('Posterior MAP failed: ' + result.message)
        w = result.x.reshape(phi.shape[1], k)
        prob = softmax(train @ w, axis=1)
        hessian = np.eye(d) / self.prior_sd**2
        for a, p in zip(train, prob):
            hessian += np.kron(np.outer(a, a), np.diag(p) - np.outer(p, p))
        lower = cholesky(hessian, lower=True)
        rng = np.random.default_rng(seed)
        draws = result.x[:, None] + solve_triangular(lower.T, rng.normal(size=(d, samples)), lower=False)
        weights = draws.T.reshape(samples, phi.shape[1], k)
        predictions = softmax(np.einsum('nf,sfc->snc', phi, weights), axis=-1)
        meta = dict(self.feature_metadata, samples=samples, map_iterations=int(result.nit),
                    map_gradient_inf=float(np.max(np.abs(result.jac))))
        return predictions, meta


def _objective(w, phi, y, classes, prior_sd):
    w = w.reshape(phi.shape[1], classes)
    z = phi @ w
    p = softmax(z, axis=1)
    value = np.sum(logsumexp(z, axis=1) - z[np.arange(len(y)), y]) + np.sum(w*w)/(2*prior_sd**2)
    p[np.arange(len(y)), y] -= 1
    grad = phi.T @ p + w/prior_sd**2
    return float(value), grad.ravel()
