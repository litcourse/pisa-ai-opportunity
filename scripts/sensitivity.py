"""Prespecified specification, context and selection sensitivity analyses."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from survey import groups, ratio, covariance, inference, equal_system
from target import target_mask

ROOT = Path(__file__).resolve().parents[1]
F = json.loads((ROOT / "config/analysis.json").read_text())
x = pd.read_parquet(ROOT / "data_processed/availability.parquet")
wc = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]
target_valid = target_mask(x, wc)
system_codes = x.CNT.to_numpy()
rows = []
est = {}
names = {}
convergence = []


def save(f, c, e, labels, n):
    assert e.shape == (81, len(labels)) and np.isfinite(e).all()
    est[f + "__" + c] = e
    names[f] = labels
    se, lo, hi, p = inference(e[0], covariance(e))
    for j, l in enumerate(labels):
        rows.append(
            dict(
                family=f,
                system=c,
                metric=l,
                estimate=e[0, j],
                se=se[j],
                ci_low=lo[j],
                ci_high=hi[j],
                p=p[j],
                n=n,
                systems=1,
            )
        )


def fit_ols(X, Y, w):
    return np.linalg.lstsq(X.T @ (w[:, None] * X), X.T @ (w[:, None] * Y), rcond=1e-12)[0]


# An independent numerical audit found separation in small immigration cells.
# Use an explicit lambda1 L2 sensitivity rather than reporting a finite unpenalized MLE.
from adjustment import fit_logit

for c in F["systems"]:
    d = x[(x.CNT == c) & x.support_sample].copy()
    xx = d.ESCS.to_numpy()
    W = d[wc].to_numpy()
    q = groups(xx, F["quartile_cuts"][c])
    y = (d.IC170Q10DA.eq(6) & d.IC171Q10DA.eq(6)).to_numpy(float)
    X = np.column_stack([np.ones(len(d)), xx])
    e = np.array([[fit_ols(X, y[:, None], w)[1, 0]] for w in W.T])
    save("continuous_SES", c, e, ["double_unavailability_slope"], len(d))
    cc = d[["AGE", "GRADE", "IMMIG"]].notna().all(axis=1).to_numpy()
    dc = d.loc[cc]
    ww = W[cc]
    qq = q[cc]
    yy = y[cc]
    covars = np.column_stack(
        [dc.AGE.to_numpy() - 15.75, dc.GRADE.to_numpy(), dc.IMMIG.eq(2), dc.IMMIG.eq(3)]
    )
    covars = covars[:, np.std(covars, axis=0) > 1e-10]
    X = np.column_stack([np.ones(len(dc)), qq == 0, qq == 1, qq == 2, covars])
    Xl = X.copy()
    Xh = X.copy()
    Xl[:, 1:4] = [1, 0, 0]
    Xh[:, 1:4] = 0
    out = []
    iterations = []
    for w in ww.T:
        raw = np.average(yy[qq == 0], weights=w[qq == 0]) - np.average(
            yy[qq == 3], weights=w[qq == 3]
        )
        lin = fit_ols(X, yy[:, None], w)[1, 0]
        b, it = fit_logit(X, yy, w)
        iterations.append(it)
        log = np.average(expit(Xl @ b) - expit(Xh @ b), weights=w)
        out.append([raw, lin, log, lin - raw, log - raw])
    save(
        "covariate_adjusted",
        c,
        np.array(out),
        [
            "complete_background_raw_RD",
            "LPM_adjusted_RD",
            "regularized_logit_standardized_RD",
            "LPM_minus_raw",
            "regularized_logit_minus_raw",
        ],
        len(dc),
    )
    convergence.append(
        {
            "system": c,
            "fits": 81,
            "max_iterations": max(iterations),
            "penalty": "L2 lambda1 exceptintercept; replicate weights normalizedmean1",
        }
    )
    # Alternative contextual model uses leave-one-student-out weighted school ESCS, excluding singleton schools.
    _, g = np.unique(d.CNTSCHID.to_numpy(), return_inverse=True)
    counts = np.bincount(g)
    ok = counts[g] > 1
    vv = []
    for w in W.T:
        sw = np.bincount(g, weights=w)
        sx = np.bincount(g, weights=w * xx)
        den = sw[g] - w
        valid = ok & (den > 1e-10)
        other = (sx[g][valid] - w[valid] * xx[valid]) / den[valid]
        xxv = xx[valid]
        Z = np.column_stack([np.ones(valid.sum()), xxv - other, other])
        yv = d.IC170Q10DA.ne(6).to_numpy(float)[valid]
        b = fit_ols(Z, yv[:, None], w[valid])
        vv.append(b[1:, 0])
    save(
        "leave_one_out_school_means",
        c,
        np.array(vv),
        ["within_context_slope", "between_context_slope"],
        int(ok.sum()),
    )
    # Partial-response identification bounds, ESCS observed ICT target; school ID not needed for a marginal proportion.
    target = (system_codes == c) & target_valid
    dt = x.iloc[np.flatnonzero(target)]
    ww = dt[wc].to_numpy()
    qq = groups(dt.ESCS.to_numpy(), F["quartile_cuts"][c])
    s = dt.IC170Q10DA
    o = dt.IC171Q10DA
    known_zero = (s.isin(range(1, 6)) | o.isin(range(1, 6))).to_numpy()
    known_one = (s.eq(6) & o.eq(6)).to_numpy()
    upper = ~known_zero
    l1 = ratio(known_one, qq == 0, ww)
    u1 = ratio(upper, qq == 0, ww)
    l4 = ratio(known_one, qq == 3, ww)
    u4 = ratio(upper, qq == 3, ww)
    save(
        "missingness_bounds",
        c,
        np.column_stack([l1, u1, l4, u4, l1 - u4, u1 - l4]),
        ["Q1_lower", "Q1_upper", "Q4_lower", "Q4_upper", "RD_lower_bound", "RD_upper_bound"],
        len(dt),
    )
    print(c, flush=True)

for f, labels in names.items():
    keys = [k for k in est if k.startswith(f + "__")]
    p, v = equal_system([est[k] for k in keys])
    se, lo, hi, pv = inference(p, v)
    for j, l in enumerate(labels):
        rows.append(
            dict(
                family=f,
                system="EQUAL_SYSTEM",
                metric=l,
                estimate=p[j],
                se=se[j],
                ci_low=lo[j],
                ci_high=hi[j],
                p=pv[j],
                n=sum(
                    next(
                        r["n"] for r in rows if r["family"] == f and r["system"] == k.split("__")[1]
                    )
                    for k in keys
                ),
                systems=len(keys),
            )
        )
main = np.load(ROOT / "outputs/qa/joint_replicates.npz")
for omit in F["systems"] + ["RESTRICTED_FRAMES"]:
    dropped = [omit] if omit != "RESTRICTED_FRAMES" else ["QCI", "QUA", "LBN"]
    cs = [c for c in F["systems"] if c not in dropped]
    p, v = equal_system([main["main__" + c][:, [6]] for c in cs])
    se, lo, hi, pv = inference(p, v)
    rows.append(
        dict(
            family="leave_systems_out",
            system=omit,
            metric="neither_RD",
            estimate=p[0],
            se=se[0],
            ci_low=lo[0],
            ci_high=hi[0],
            p=pv[0],
            n=np.nan,
            systems=len(cs),
        )
    )
pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/sensitivities.csv", index=False)
np.savez_compressed(ROOT / "outputs/qa/sensitivity_replicates.npz", **est)
(ROOT / "outputs/qa/sensitivity_manifest.json").write_text(
    json.dumps(
        {
            "names": names,
            "logit_convergence": convergence,
            "missing_bounds": "point identification bounds, not confidence intervals; individual bound endpoints have BRR uncertainty separately; target ESCS-known ICT students using complete-case cutpoints",
            "context": "Leave-one-out is a contextual regression sensitivity, not the exact within/between covariance decomposition.",
        },
        indent=2,
    )
)
print(pd.DataFrame(rows).query("system == 'EQUAL_SYSTEM'").to_string(index=False))
