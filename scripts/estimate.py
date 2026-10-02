"""Execute frozen primary analyses; retain complete joint BRR vectors."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import json
import numpy as np
import pandas as pd
from survey import (
    groups,
    ratio,
    covariance,
    inference,
    equal_system,
    holm,
    joint,
    decomposition,
    quantiles,
)

ROOT = Path(__file__).resolve().parents[1]
freeze = json.loads((ROOT / "config/analysis.json").read_text())
assert freeze["analysis_frozen"]
x = pd.read_parquet(ROOT / "data_processed/availability.parquet")
wc = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]
rows = []
est = {}
names = {}
checks = []


def keep(family, system, e, labels, n, schools):
    assert e.shape == (81, len(labels)) and np.isfinite(e).all()
    est[f"{family}__{system}"] = e
    names[family] = labels
    p = e[0]
    v = covariance(e)
    se, lo, hi, pv = inference(p, v)
    for j, label in enumerate(labels):
        rows.append(
            dict(
                family=family,
                system=system,
                metric=label,
                estimate=p[j],
                se=se[j],
                ci_low=lo[j],
                ci_high=hi[j],
                p=pv[j],
                n=n,
                schools=schools,
                systems=1,
            )
        )


def opportunity(d, W, q):
    a = d.IC170Q10DA.ne(6).to_numpy()
    b = d.IC171Q10DA.ne(6).to_numpy()
    states = [a & b, a & ~b, ~a & b, ~a & ~b]
    vals = [ratio(s, np.ones(len(d), bool), W) for s in states]
    labels = ["both", "school_only", "outside_only", "neither"]
    for name, y in [("neither", states[3]), ("school_available", a), ("outside_available", b)]:
        low = ratio(y, q == 0, W)
        high = ratio(y, q == 3, W)
        vals += [low, high, low - high]
        labels += [f"{name}_Q1", f"{name}_Q4", f"{name}_RD"]
    return np.column_stack(vals), labels


for c in freeze["systems"]:
    d = x[(x.CNT == c) & x.support_sample].copy()
    W = d[wc].to_numpy()
    q = groups(d.ESCS.to_numpy(), freeze["quartile_cuts"][c])
    ns = d.CNTSCHID.nunique()
    e, labels = opportunity(d, W, q)
    keep("main", c, e, labels, len(d), ns)
    checks.append(float(np.max(np.abs(e[:, :4].sum(axis=1) - 1))))
    if c in freeze["conditional_systems"]:
        y = d.IC170Q10DA.ne(6).to_numpy()
        o = d.IC171Q10DA.eq(6).to_numpy()
        allq = ratio(y, o, W)
        lo = ratio(y, o & (q == 0), W)
        hi = ratio(y, o & (q == 3), W)
        keep(
            "conditional",
            c,
            np.column_stack([allq, lo, hi, lo - hi]),
            [
                "school_available_given_outside_unavailable",
                "conditional_Q1",
                "conditional_Q4",
                "conditional_RD",
            ],
            len(d),
            ns,
        )
    cm = (
        d[["IC170Q01JA", "IC171Q01JA", "IC170Q04JA", "IC171Q04JA"]]
        .isin(range(1, 7))
        .all(axis=1)
        .to_numpy()
    )
    dc = d.loc[cm]
    WW = W[cm]
    qq = q[cm]
    if c in freeze["common_systems"]:
        vv = []
        for s, o in [
            ("IC170Q10DA", "IC171Q10DA"),
            ("IC170Q01JA", "IC171Q01JA"),
            ("IC170Q04JA", "IC171Q04JA"),
        ]:
            y = (dc[s].eq(6) & dc[o].eq(6)).to_numpy()
            vv.append(ratio(y, qq == 0, WW) - ratio(y, qq == 3, WW))
        ee = np.column_stack(vv + [vv[0] - vv[1], vv[0] - vv[2]])
        keep(
            "tools",
            c,
            ee,
            ["AI_RD", "computer_RD", "internet_RD", "AI_minus_computer_RD", "AI_minus_internet_RD"],
            len(dc),
            dc.CNTSCHID.nunique(),
        )
    xx = d.ESCS.to_numpy()
    y = d.IC170Q10DA.ne(6).to_numpy(float)
    e = decomposition(xx, y, d.CNTSCHID.to_numpy(), W)
    keep(
        "decomposition",
        c,
        e,
        [
            "total_slope",
            "within_slope",
            "between_slope",
            "within_contribution",
            "between_contribution",
            "total_minus_within",
        ],
        len(d),
        ns,
    )
    checks.append(float(np.max(np.abs(e[:, 3] + e[:, 4] - e[:, 0]))))
    # Same complete-three-tool sample on all40: sample-composition check, separate from supported37 comparator family.
    e, labels = opportunity(dc, WW, qq)
    keep("common_sample_sensitivity", c, e, labels, len(dc), dc.CNTSCHID.nunique())
    v = []
    for j in range(81):
        qr = groups(xx, quantiles(xx, W[:, j]))
        yy = d.IC170Q10DA.eq(6).to_numpy() & d.IC171Q10DA.eq(6).to_numpy()
        low = np.average(yy[qr == 0], weights=W[qr == 0, j])
        high = np.average(yy[qr == 3], weights=W[qr == 3, j])
        v.append([low - high])
    keep("recomputed_quartiles", c, np.array(v), ["neither_RD"], len(d), ns)
    for min_n in [5, 10]:
        m = d.CNTSCHID.map(d.CNTSCHID.value_counts()).gt(min_n).to_numpy()
        ee = decomposition(xx[m], y[m], d.CNTSCHID.to_numpy()[m], W[m])
        keep(
            f"school_gt{min_n}",
            c,
            ee,
            names["decomposition"],
            int(m.sum()),
            d.loc[m, "CNTSCHID"].nunique(),
        )
    print(c, len(d), flush=True)

summaries = {}
for family, labels in names.items():
    keys = [k for k in est if k.startswith(family + "__")]
    pp, v = equal_system([est[k] for k in keys])
    se, lo, hi, pv = inference(pp, v)
    summaries[family] = {
        "metrics": labels,
        "point": pp.tolist(),
        "covariance": v.tolist(),
        "systems": len(keys),
    }
    for j, label in enumerate(labels):
        rows.append(
            dict(
                family=family,
                system="EQUAL_SYSTEM",
                metric=label,
                estimate=pp[j],
                se=se[j],
                ci_low=lo[j],
                ci_high=hi[j],
                p=pv[j],
                n=sum(
                    next(
                        r["n"]
                        for r in rows
                        if r["family"] == family and r["system"] == k.split("__")[1]
                    )
                    for k in keys
                ),
                schools=np.nan,
                systems=len(keys),
            )
        )
    np.save(ROOT / "outputs/qa" / f"{family}_summary_cov.npy", v)
tool = summaries["tools"]
p = np.array(tool["point"])[3:]
cov = np.array(tool["covariance"])[3:, 3:]
pv = inference(p, cov)[3]
stat, df, jp = joint(p, cov)
test = {
    "contrasts": tool["metrics"][3:],
    "unadjusted_p": pv.tolist(),
    "holm_p": holm(pv).tolist(),
    "joint_wald": stat,
    "joint_df": df,
    "joint_p": jp,
    "systems": tool["systems"],
}
pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/estimates.csv", index=False)
np.savez_compressed(ROOT / "outputs/qa/joint_replicates.npz", **est)
(ROOT / "outputs/qa/result_manifest.json").write_text(
    json.dumps(
        {
            "names": names,
            "summaries": summaries,
            "secondary_test": test,
            "max_identity_error": max(checks),
        },
        indent=2,
    )
)
print(
    json.dumps(
        {
            "primary": summaries["main"],
            "secondary": test,
            "decomposition": summaries["decomposition"],
        },
        indent=2,
    )
)
