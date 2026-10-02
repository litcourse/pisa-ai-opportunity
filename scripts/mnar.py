"""Post-result pattern-mixture sensitivity with partial outcomes held fixed.

Not a MAR test, preregistration, imputation draw, or causal analysis.
See README.md for the prior-to-execution choices.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import brentq
from survey import groups, covariance, inference
from adjustment import fit_logit
from target import target_mask

ROOT = Path(__file__).resolve().parents[1]
DELTAS = np.arange(-2.0, 2.01, 0.5)
PROFILES = [(float(a), float(b)) for a in DELTAS for b in DELTAS]


def partial_outcomes(s, o):
    valid_s = (s >= 1) & (s <= 6)
    valid_o = (o >= 1) & (o <= 6)
    known_one = (s == 6) & (o == 6)
    known_zero = ((s >= 1) & (s <= 5)) | ((o >= 1) & (o <= 5))
    patterns = [(s == 6) & ~valid_o, (o == 6) & ~valid_s, ~valid_s & ~valid_o]
    assert np.all(known_one | known_zero | np.logical_or.reduce(patterns))
    return known_one, known_zero, patterns, valid_s & valid_o


def expected_outcome(known_one, known_zero, anchor, delta):
    """Deterministic expected binary outcome; never overwrite observed outcomes."""
    ambiguous = ~(known_one | known_zero)
    y = known_one.astype(float)
    y[ambiguous] = expit(logit(anchor[ambiguous]) + delta)
    return y


def predictors(d, q):
    cols = [np.ones(len(d))] + [(q == k).astype(float) for k in range(3)]
    for name in ["AGE", "GRADE"]:
        z = d[name].to_numpy(float)
        missing = ~np.isfinite(z)
        median = np.nanmedian(z)
        if not np.isfinite(median):
            median = 0.0
        z = np.where(missing, median, z)
        z = (z - z.mean()) / (z.std() or 1.0)
        cols += [z, missing.astype(float)]
    cols += [
        d.IMMIG.eq(2).to_numpy(float),
        d.IMMIG.eq(3).to_numpy(float),
        d.IMMIG.isna().to_numpy(float),
    ]
    X = np.column_stack(cols)
    return X[:, (np.std(X, axis=0) > 1e-10) | (np.arange(X.shape[1]) == 0)]


def risk_for_delta(cache, quartile, delta):
    q, one, zero, A, W = cache
    m = q == quartile
    ambiguous = m & ~(one | zero)
    den = W[m].sum(axis=0)
    if np.any(den <= 0):
        raise ValueError("Empty target SES denominator")
    fixed = W[m & one].sum(axis=0)
    predicted = (W[ambiguous] * expit(logit(A[ambiguous]) + delta)).sum(axis=0)
    return (fixed + predicted) / den


def aggregate_profile(caches, delta1, delta4):
    vectors = [risk_for_delta(c, 0, delta1) - risk_for_delta(c, 3, delta4) for c in caches.values()]
    point = float(np.mean([e[0] for e in vectors]))
    var = sum(float(covariance(e[:, None])[0, 0]) for e in vectors) / len(vectors) ** 2
    se = np.sqrt(var)
    return point, se, point - 1.95996398454 * se, point + 1.95996398454 * se


def main():
    frozen = json.loads((ROOT / "config/analysis.json").read_text())
    x = pd.read_parquet(ROOT / "data_processed/availability.parquet")
    wc = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]
    good = target_mask(x, wc)
    codes = x.CNT.to_numpy()
    caches = {}
    support = []
    grid_rows = []
    fit_rows = []
    saved = {}
    for system in frozen["systems"]:
        d = x.iloc[np.flatnonzero(good & (codes == system))]
        q = groups(d.ESCS.to_numpy(), frozen["quartile_cuts"][system])
        s = d.IC170Q10DA.to_numpy()
        o = d.IC171Q10DA.to_numpy()
        one, zero, patterns, full = partial_outcomes(s, o)
        W = d[wc].to_numpy(float)
        X = predictors(d, q)
        train = [full & (s == 6), full & (o == 6), full]
        outcomes = [o == 6, s == 6, one]
        if any(not t.any() or np.unique(y[t]).size < 2 for t, y in zip(train, outcomes)):
            raise ValueError(f"Unsupported outcome reference {system}")
        A = np.full((len(d), 81), 0.5)
        for j in range(81):
            for k in range(3):
                t = train[k]
                b, iterations = fit_logit(X[t], outcomes[k][t].astype(float), W[t, j])
                A[patterns[k], j] = expit(X[patterns[k]] @ b)
                if j == 0:
                    fit_rows.append(
                        dict(
                            system=system,
                            model=k,
                            train_n=int(t.sum()),
                            events=int(outcomes[k][t].sum()),
                            iterations=iterations,
                        )
                    )
        if np.any((A <= 0) | (A >= 1)):
            raise ValueError("Degenerate predicted probability")
        cache = (q, one, zero, A, W)
        caches[system] = cache
        for quartile in [0, 3]:
            m = q == quartile
            w = W[:, 0]
            den = w[m].sum()
            row = dict(
                system=system,
                quartile=quartile + 1,
                target_n=int(m.sum()),
                known_one=float(w[m & one].sum() / den),
                known_zero=float(w[m & zero].sum() / den),
            )
            for k, pattern in enumerate(patterns):
                row[f"ambiguous_pattern{k}"] = float(w[m & pattern].sum() / den)
            row["anchor_risk"] = float(risk_for_delta(cache, quartile, 0)[0])
            support.append(row)
        E = np.column_stack(
            [risk_for_delta(cache, 0, a) - risk_for_delta(cache, 3, b) for a, b in PROFILES]
        )
        saved[system] = E
        for k, (a, b) in enumerate(PROFILES):
            point = E[0, k]
            se = np.sqrt(covariance(E[:, k : k + 1])[0, 0])
            grid_rows.append(
                dict(
                    system=system,
                    delta_q1=a,
                    delta_q4=b,
                    estimate=point,
                    se=se,
                    ci_low=point - 1.95996398454 * se,
                    ci_high=point + 1.95996398454 * se,
                )
            )
        print(system, flush=True)
    for a, b in PROFILES:
        p, se, lo, hi = aggregate_profile(caches, a, b)
        grid_rows.append(
            dict(
                system="EQUAL_SYSTEM",
                delta_q1=a,
                delta_q4=b,
                estimate=p,
                se=se,
                ci_low=lo,
                ci_high=hi,
            )
        )
    tipping = []
    for name, index in [("point_zero", 0), ("lower_ci_zero", 2), ("upper_ci_zero", 3)]:

        def f(z):
            return aggregate_profile(caches, 0.0, z)[index]

        scan = np.linspace(-12, 12, 49)
        brackets = [(a, b) for a, b in zip(scan[:-1], scan[1:]) if f(a) * f(b) <= 0]
        if brackets:
            delta = brentq(f, *brackets[0], xtol=1e-10)
            p, se, lo, hi = aggregate_profile(caches, 0.0, delta)
            tipping.append(
                dict(
                    threshold=name,
                    delta_q1=0.0,
                    delta_q4=delta,
                    outcome_odds_multiplier_q4=float(np.exp(delta)),
                    estimate=p,
                    se=se,
                    ci_low=lo,
                    ci_high=hi,
                )
            )
        else:
            tipping.append(dict(threshold=name, status="no crossing in [-12,12]"))
    pd.DataFrame(grid_rows).to_csv(ROOT / "outputs/tables/MNAR_grid.csv", index=False)
    pd.DataFrame(tipping).to_csv(ROOT / "outputs/tables/MNAR_tipping.csv", index=False)
    pd.DataFrame(support).to_csv(ROOT / "outputs/tables/MNAR_patterns.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(ROOT / "outputs/tables/MNAR_fits.csv", index=False)
    np.savez_compressed(ROOT / "outputs/qa/MNAR_replicates.npz", **saved)
    # Anchors are analysis-internal individual-level predictions: never add to sharing package.
    np.savez_compressed(
        ROOT / "data_processed/MNAR_internal.npz",
        **{
            f"{c}_{k}": v
            for c, cache in caches.items()
            for k, v in zip(["q", "one", "zero", "anchor", "W"], cache)
        },
    )
    report = dict(
        status="exploratory post-result pattern-mixture sensitivity",
        target_n=sum(len(c[0]) for c in caches.values()),
        extreme_quartile_n=sum(r["target_n"] for r in support),
        systems=len(caches),
        anchor_profile=aggregate_profile(caches, 0.0, 0.0),
        delta_grid=DELTAS.tolist(),
        tipping=tipping,
        protocol="README.md",
        interpretation="delta is outcome odds shift among ambiguous outcomes; model conditional, not a response odds ratio; unrestricted bounds remain",
    )
    (ROOT / "outputs/qa/MNAR_manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
