"""Weighted ridge logistic sensitivity: lambda=1, no intercept penalty."""

import numpy as np
from scipy.special import expit


def fit_logit(X, y, w, penalty=1.0):
    w = w / w.mean()
    b = np.zeros(X.shape[1])
    ym = np.clip(np.average(y, weights=w), 1e-8, 1 - 1e-8)
    b[0] = np.log(ym / (1 - ym))
    R = np.eye(X.shape[1]) * penalty
    R[0, 0] = 0

    def ll(z):
        a = X @ z
        return np.sum(w * (y * a - np.logaddexp(0, a))) - 0.5 * z @ R @ z

    for it in range(100):
        p = expit(X @ b)
        grad = X.T @ (w * (y - p)) - R @ b
        H = X.T @ ((w * np.maximum(p * (1 - p), 1e-12))[:, None] * X) + R
        delta = np.linalg.solve(H, grad)
        step = 1.0
        old = ll(b)
        while step > 1e-6 and ll(b + step * delta) < old - 1e-8:
            step /= 2
        b = b + step * delta
        if np.max(np.abs(step * delta)) < 1e-7:
            if np.max(np.abs(X.T @ (w * (y - expit(X @ b))) - R @ b)) > 1e-4:
                raise ValueError("Score did not converge")
            return b, it + 1
    raise ValueError("Penalized logit did not converge")
