"""Explicit ST004D01T sensitivity on systems that actually release this field."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import json
import pandas as pd
import numpy as np
from survey import groups, ratio, covariance, inference, equal_system

ROOT = Path(__file__).resolve().parents[1]
F = json.loads((ROOT / "config/analysis.json").read_text())
x = pd.read_parquet(ROOT / "data_processed/availability.parquet")
g = pd.read_parquet(
    ROOT / "data_processed/student_items.parquet", columns=["CNT", "CNTSTUID", "ST004D01T"]
)
x = x.merge(g, on=["CNT", "CNTSTUID"], validate="one_to_one")
wc = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]
rows = []
audit = []
est = {}
labels = [
    "raw_RD",
    "age_grade_immig_adjusted_RD",
    "also_ST004_adjusted_RD",
    "sex_adjustment_change",
]
for c in F["systems"]:
    base = x[(x.CNT == c) & x.support_sample]
    d = base[
        base.ST004D01T.isin([1, 2]) & base[["AGE", "GRADE", "IMMIG"]].notna().all(axis=1)
    ].copy()
    audit.append(
        dict(
            system=c,
            core_n=len(base),
            ST004_valid_n=int(base.ST004D01T.isin([1, 2]).sum()),
            complete_adjustment_n=len(d),
        )
    )
    if len(d) < 500 or d.CNTSCHID.nunique() < 50:
        continue
    q = groups(d.ESCS.to_numpy(), F["quartile_cuts"][c])
    W = d[wc].to_numpy()
    y = (d.IC170Q10DA.eq(6) & d.IC171Q10DA.eq(6)).to_numpy(float)
    covars = np.column_stack(
        [d.AGE.to_numpy() - 15.75, d.GRADE.to_numpy(), d.IMMIG.eq(2), d.IMMIG.eq(3)]
    )
    covars = covars[:, covars.std(axis=0) > 1e-10]
    X = np.column_stack([np.ones(len(d)), q == 0, q == 1, q == 2, covars])
    Z = np.column_stack([X, d.ST004D01T.eq(2)])
    out = []
    for j, w in enumerate(W.T):
        raw = np.average(y[q == 0], weights=w[q == 0]) - np.average(y[q == 3], weights=w[q == 3])
        b = np.linalg.lstsq(X.T @ (w[:, None] * X), X.T @ (w * y), rcond=1e-12)[0][1]
        z = np.linalg.lstsq(Z.T @ (w[:, None] * Z), Z.T @ (w * y), rcond=1e-12)[0][1]
        out.append([raw, b, z, z - b])
    e = np.array(out)
    est[c] = e
    se, lo, hi, p = inference(e[0], covariance(e))
    for j, l in enumerate(labels):
        rows.append(
            dict(
                system=c,
                metric=l,
                estimate=e[0, j],
                se=se[j],
                ci_low=lo[j],
                ci_high=hi[j],
                p=p[j],
                n=len(d),
                systems=1,
            )
        )
p, v = equal_system(list(est.values()))
se, lo, hi, pv = inference(p, v)
for j, l in enumerate(labels):
    rows.append(
        dict(
            system="EQUAL_SYSTEM",
            metric=l,
            estimate=p[j],
            se=se[j],
            ci_low=lo[j],
            ci_high=hi[j],
            p=pv[j],
            n=sum(r["complete_adjustment_n"] for r in audit if r["system"] in est),
            systems=len(est),
        )
    )
pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/ST004_sensitivity.csv", index=False)
pd.DataFrame(audit).to_csv(ROOT / "outputs/tables/ST004_coverage.csv", index=False)
np.savez_compressed(ROOT / "outputs/qa/ST004_replicates.npz", **est)
print(pd.DataFrame(rows).query("system == 'EQUAL_SYSTEM'").to_string(index=False))
