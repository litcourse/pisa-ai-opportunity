"""Additional post-primary MAR sensitivity, not a confirmatory test."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from survey import groups, ratio, covariance, inference, equal_system, kish
from adjustment import fit_logit
from target import target_mask

ROOT = Path(__file__).resolve().parents[1]
F = json.loads((ROOT / "config/analysis.json").read_text())
x = pd.read_parquet(ROOT / "data_processed/availability.parquet")
wc = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]
target_valid = target_mask(x, wc)
system_codes = x.CNT.to_numpy()
rows = []
est = {}
support = []
for c in F["systems"]:
    target = (system_codes == c) & target_valid
    d = x.iloc[np.flatnonzero(target)]
    W = d[wc].to_numpy()
    R = d.support_sample.to_numpy()
    q = groups(d.ESCS.to_numpy(), F["quartile_cuts"][c])
    z = d.ESCS.to_numpy()
    z = (z - np.mean(z)) / np.std(z)
    cov = [np.ones(len(d)), z, z * z]
    for name in ["AGE", "GRADE"]:
        val = d[name].to_numpy()
        miss = ~np.isfinite(val)
        val = np.where(miss, np.nanmedian(val), val)
        val = (val - val.mean()) / (val.std() or 1.0)
        cov += [val, miss.astype(float)]
    cov += [
        d.IMMIG.eq(2).to_numpy(float),
        d.IMMIG.eq(3).to_numpy(float),
        d.IMMIG.isna().to_numpy(float),
    ]
    X = np.column_stack(cov)
    X = X[:, (np.std(X, axis=0) > 1e-10) | np.r_[True, np.zeros(X.shape[1] - 1, bool)]]
    y = (d.IC170Q10DA.eq(6) & d.IC171Q10DA.eq(6)).to_numpy()
    yy = y[R]
    qq = q[R]
    out = []
    diags = []
    for j in range(81):
        w = W[:, j]
        b, it = fit_logit(X, R.astype(float), w)
        p = expit(X @ b)
        clipped = np.clip(p, 0.05, 0.95)
        iw = w[R] / clipped[R]
        lo = np.average(yy[qq == 0], weights=iw[qq == 0])
        hi = np.average(yy[qq == 3], weights=iw[qq == 3])
        out.append([lo - hi])
        if j == 0:
            diags = [
                float(np.min(p)),
                float(np.max(p)),
                float(np.average((p < 0.05) | (p > 0.95), weights=w)),
                kish(iw),
                float(np.average(R, weights=w)),
            ]
    e = np.array(out)
    est[c] = e
    se, lo, hi, pv = inference(e[0], covariance(e))
    rows.append(
        dict(
            system=c,
            estimate=e[0, 0],
            se=se[0],
            ci_low=lo[0],
            ci_high=hi[0],
            p=pv[0],
            target_n=len(d),
            observed_n=int(R.sum()),
        )
    )
    support.append(
        dict(
            system=c,
            propensity_min=diags[0],
            propensity_max=diags[1],
            clipped_weighted_fraction=diags[2],
            ipw_kish=diags[3],
            weighted_complete_fraction=diags[4],
        )
    )
    print(c, flush=True)
p, v = equal_system(list(est.values()))
se, lo, hi, pv = inference(p, v)
rows.append(
    dict(
        system="EQUAL_SYSTEM",
        estimate=p[0],
        se=se[0],
        ci_low=lo[0],
        ci_high=hi[0],
        p=pv[0],
        target_n=sum(r["target_n"] for r in rows),
        observed_n=sum(r["observed_n"] for r in rows),
    )
)
pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/MAR_selection_sensitivity.csv", index=False)
pd.DataFrame(support).to_csv(ROOT / "outputs/tables/selection_overlap.csv", index=False)
np.savez_compressed(ROOT / "outputs/qa/selection_replicates.npz", **est)
(ROOT / "outputs/qa/selection_assumptions.json").write_text(
    json.dumps(
        {
            "status": "additional post-primary sensitivity; no confirmatory p claim",
            "model": "response to bothAI plus school/designeligibility ~ ESCS standardized+square, age, grade, immigration dummies; missing covariates imputedmedian withmissingindicator; L2lambda1 exceptintercept; predictorstandardization heldfixed overreplicates",
            "propensity_clip": [0.05, 0.95],
            "assumption": "Selection ignorable conditional on included covariates, propensity model adequate and positivity; cannot resolve nonignorable response or ESCSmissing; schoolID selection mayviolate model",
            "target": "ESCS-known ICTstudents in40 frozen systems; completecase SES cutpoints fixed",
            "result": rows[-1],
        },
        indent=2,
    )
)
print(rows[-1])
