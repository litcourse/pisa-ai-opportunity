"""Direct, non-PV survey estimators for the AI opportunity study."""

import numpy as np
from scipy.stats import norm, chi2


def quantiles(x, w):
    order = np.argsort(x, kind="stable")
    return np.interp([0.25, 0.5, 0.75], np.cumsum(w[order]) / w.sum(), x[order])


def groups(x, cuts):
    return np.searchsorted(cuts, x, side="left")


def ratio(y, mask, W):
    den = W.T @ mask.astype(float)
    if np.any(den <= 0):
        raise ValueError("Nonpositive replicate denominator")
    return (W.T @ (y * mask)) / den


def covariance(estimates):
    d = estimates[1:] - estimates[0]
    return d.T @ d / 20.0


def inference(point, cov):
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    z = np.divide(point, se, out=np.full_like(point, np.nan), where=se > 0)
    return se, point - 1.95996398454 * se, point + 1.95996398454 * se, 2 * norm.sf(np.abs(z))


def equal_system(estimates):
    point = np.mean([e[0] for e in estimates], axis=0)
    cov = sum(covariance(e) for e in estimates) / len(estimates) ** 2
    return point, cov


def holm(p):
    order = np.argsort(p)
    result = np.empty(len(p))
    prev = 0.0
    for i, j in enumerate(order):
        prev = max(prev, min(1.0, (len(p) - i) * p[j]))
        result[j] = prev
    return result


def joint(point, cov):
    rank = np.linalg.matrix_rank(cov)
    stat = float(point @ np.linalg.pinv(cov) @ point)
    return stat, int(rank), float(chi2.sf(stat, rank))


def decomposition(x, y, school, W):
    _, g = np.unique(school, return_inverse=True)
    out = []
    for w in W.T:
        sw = np.bincount(g, weights=w)
        sx = np.bincount(g, weights=w * x)
        sy = np.bincount(g, weights=w * y)
        mx = np.divide(sx, sw, out=np.zeros_like(sx), where=sw > 0)
        my = np.divide(sy, sw, out=np.zeros_like(sy), where=sw > 0)
        xc = x - np.average(x, weights=w)
        yc = y - np.average(y, weights=w)
        vt = np.sum(w * xc * xc)
        ct = np.sum(w * xc * yc)
        vw = np.sum(w * (x - mx[g]) ** 2)
        cw = np.sum(w * (x - mx[g]) * (y - my[g]))
        vb = vt - vw
        cb = ct - cw
        out.append([ct / vt, cw / vw, cb / vb, cw / vt, cb / vt, (ct / vt) - (cw / vw)])
    return np.array(out)


def kish(w):
    return float(w.sum() ** 2 / (w @ w))
