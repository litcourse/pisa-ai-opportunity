"""Exploratory matched-domain non-device SES proxy sensitivity; never an ESCS replacement."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import datetime, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from survey import quantiles, groups, decomposition, covariance, equal_system

ROOT = Path(__file__).resolve().parents[1]
F = json.loads((ROOT / "config/analysis.json").read_text())
wc = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]
cols = [
    "CNT",
    "CNTSTUID",
    "CNTSCHID",
    "ESCS",
    "support_sample",
    "IC170Q10DA",
    "IC171Q10DA",
    "IC170Q01JA",
    "IC171Q01JA",
    "IC170Q04JA",
    "IC171Q04JA",
] + wc
x = pd.read_parquet(
    ROOT / "data_processed/availability.parquet",
    columns=cols,
    filters=[("support_sample", "=", True)],
)
p = pd.read_parquet(
    ROOT / "data_processed/parental_background.parquet",
    columns=["CNT", "CNTSTUID", "HISEI", "HISCED"],
)
x = x.merge(p, on=["CNT", "CNTSTUID"], validate="one_to_one")
coverage = []
prepared = {}
protocol = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "stage": "exploratory supplementary analysis after original effects visible",
    "parent_population": "original40-system243364 paired complete domain; additional proxy validity creates matched domain",
    "HISEI": "highest parental occupational status; independent of digital possessions in its construction, but an ESCS component and not external criterion",
    "HISCED": "highest parental education categories; low1–4 throughuppersecondary, high7–9 bachelororhigher; categories5/6 retained in matchedwhole-sample continuous/decomposition summaries; no randomtie splitting",
    "baseline": "original frozen ESCS quartile cuts on identical records, not different samples",
    "HISEI_group_rule": "weighted cumulative interpolated proxy quartiles; cutties in lowergroup; cutpoints fixed overreplicates",
    "school_scale": "each SES variable standardized by matched-domain mainweight mean/SD; constants heldfixed overreplicates; comparative unit oneSD",
    "main_support": ">=500 matchedvalidstudents/50schools and positive low/high denominators in all81weights; Kish<30/10schools is precisionflag",
    "tools_support": "startoriginal37toolsystems; matched3toolvalid n>=500/50schools; bothproxy andESCS low/high>=30schools andKish>=500",
    "analysis": "all80BRRFay fullpaired covariance; fixedeligible equal-system aggregation; estimates exploratory not confirmatory",
    "sources": json.loads(
        (ROOT / "outputs/qa/parental_field_audit.json").read_text(encoding="utf-8")
    ),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}


def masks(d, proxy, cuts=None):
    e = groups(d.ESCS.to_numpy(), F["quartile_cuts"][d.CNT.iloc[0]])
    if proxy == "HISEI":
        q = groups(d.HISEI.to_numpy(), cuts)
        lo = q == 0
        hi = q == 3
    else:
        lo = d.HISCED.le(4).to_numpy()
        hi = d.HISCED.ge(7).to_numpy()
    return [(e == 0, e == 3), (lo, hi)]


def kish(w):
    return float(w.sum() ** 2 / (w @ w)) if len(w) else 0.0


for proxy in ["HISEI", "HISCED"]:
    for c in F["systems"]:
        d = x[(x.CNT == c) & x[proxy].notna()]
        d = d[np.isfinite(d[proxy].to_numpy())]
        if proxy == "HISCED":
            d = d[d.HISCED.isin(range(1, 10))]
        W = d[wc].to_numpy()
        cuts = (
            quantiles(d.HISEI.to_numpy(), d.W_FSTUWT.to_numpy())
            if proxy == "HISEI" and len(d)
            else None
        )
        overall = len(d) >= 500 and d.CNTSCHID.nunique() >= 50
        pair = masks(d, proxy, cuts) if len(d) else [(np.array([], bool), np.array([], bool))] * 2
        main_ok = overall and all(np.all(W[a].sum(axis=0) > 0) for pp in pair for a in pp)
        common = (
            d[["IC170Q01JA", "IC171Q01JA", "IC170Q04JA", "IC171Q04JA"]]
            .isin(range(1, 7))
            .all(axis=1)
            .to_numpy()
        )
        toolok = (
            c in F["common_systems"]
            and common.sum() >= 500
            and d.loc[common, "CNTSCHID"].nunique() >= 50
        )
        row = {
            "proxy": proxy,
            "system": c,
            "base_core_n": int(((x.CNT == c) & x.support_sample).sum()),
            "matched_n": len(d),
            "matched_schools": d.CNTSCHID.nunique(),
            "main_supported": bool(main_ok),
            "tools_n": int(common.sum()),
        }
        for name, pp in zip(["ESCS", "proxy"], pair):
            for g, a in zip(["low", "high"], pp):
                z = d[a]
                w = z.W_FSTUWT.to_numpy()
                row[f"{name}_{g}_n"] = len(z)
                row[f"{name}_{g}_kish"] = kish(w)
                row[f"{name}_{g}_schools"] = z.CNTSCHID.nunique()
                z = d[a & common]
                w = z.W_FSTUWT.to_numpy()
                toolok &= (
                    len(z) > 0
                    and z.CNTSCHID.nunique() >= 30
                    and kish(w) >= 500
                    and bool(np.all(z[wc].sum().to_numpy() > 0))
                )
        row["tool_supported"] = bool(toolok)
        coverage.append(row)
        if main_ok:
            prepared[(proxy, c)] = (d, W, cuts, pair, common, bool(toolok))
pd.DataFrame(coverage).to_csv(ROOT / "outputs/tables/SES_support.csv", index=False)
protocol["systems"] = {
    proxy: {
        "main": [c for (p, c) in prepared if p == proxy],
        "tools": [c for (p, c), v in prepared.items() if p == proxy and v[-1]],
    }
    for proxy in ["HISEI", "HISCED"]
}
protocol["HISEI_cuts"] = {c: v[2].tolist() for (p, c), v in prepared.items() if p == "HISEI"}
(ROOT / "outputs/qa/SES_protocol.json").write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
)
# Selection and proxy coding above fixed before any revised association.
vectors = {}
names = {}
rows = []
sizes = {}


def keep(family, system, e, labels, n):
    assert e.shape == (81, len(labels)) and np.isfinite(e).all()
    vectors[family + "__" + system] = e
    names[family] = labels
    sizes[(family, system)] = n
    U = covariance(e)
    se = np.sqrt(np.maximum(U.diagonal(), 0))
    for j, l in enumerate(labels):
        rows.append(
            dict(
                family=family,
                system=system,
                metric=l,
                estimate=e[0, j],
                se=se[j],
                ci_low=e[0, j] - 1.959963984540054 * se[j],
                ci_high=e[0, j] + 1.959963984540054 * se[j],
                p_exploratory=float(2 * norm.sf(abs(e[0, j] / se[j]))) if se[j] > 0 else np.nan,
                n=n,
                systems=1,
            )
        )


def risks(y, W, pair):
    out = []
    for lo, hi in pair:
        low = W[lo & y].sum(axis=0) / W[lo].sum(axis=0)
        high = W[hi & y].sum(axis=0) / W[hi].sum(axis=0)
        out.append(low - high)
    return out


for (proxy, c), (d, W, cuts, pair, common, toolok) in prepared.items():
    y = (d.IC170Q10DA.eq(6) & d.IC171Q10DA.eq(6)).to_numpy()
    r = risks(y, W, pair)
    keep(
        proxy + "_main",
        c,
        np.column_stack(r + [r[1] - r[0]]),
        ["matched_ESCS_RD", "proxy_RD", "proxy_minus_ESCS_RD"],
        len(d),
    )
    w = W[:, 0]
    soc = []
    for v in [d.ESCS.to_numpy(), d[proxy].to_numpy()]:
        mu = np.average(v, weights=w)
        sd = np.sqrt(np.average((v - mu) ** 2, weights=w))
        assert sd > 0
        soc.append((v - mu) / sd)
    # Matched, fixed-standardized continuous slopes; HISCED score requires ordinal spacing assumption.
    out = []
    for wr in W.T:
        outrow = []
        for v in soc:
            vc = v - np.average(v, weights=wr)
            yc = y - np.average(y, weights=wr)
            outrow.append(np.sum(wr * vc * yc) / np.sum(wr * vc * vc))
        out.append(outrow + [outrow[1] - outrow[0]])
    keep(
        proxy + "_continuous",
        c,
        np.array(out),
        ["matched_ESCS_slope_per_SD", "proxy_slope_per_SD", "proxy_minus_ESCS_slope"],
        len(d),
    )
    schooly = d.IC170Q10DA.ne(6).to_numpy(float)
    school = d.CNTSCHID.to_numpy()
    A = decomposition(soc[0], schooly, school, W)
    B = decomposition(soc[1], schooly, school, W)
    labels = [
        "total_slope",
        "within_slope",
        "between_slope",
        "within_contribution",
        "between_contribution",
        "total_minus_within",
    ]
    keep(
        proxy + "_schools",
        c,
        np.column_stack([A, B, B - A]),
        ["matched_ESCS_" + l for l in labels]
        + ["proxy_" + l for l in labels]
        + ["proxy_minus_ESCS_" + l for l in labels],
        len(d),
    )
    if toolok:
        dc = d.loc[common]
        WW = W[common]
        pp = [(a[common], b[common]) for a, b in pair]
        r = []
        for s, o in [
            ("IC170Q10DA", "IC171Q10DA"),
            ("IC170Q01JA", "IC171Q01JA"),
            ("IC170Q04JA", "IC171Q04JA"),
        ]:
            r.extend(risks((dc[s].eq(6) & dc[o].eq(6)).to_numpy(), WW, pp))
        # Order AI_E,AI_P,computer_E,computer_P,internet_E,internet_P.
        contrasts = [r[0] - r[2], r[0] - r[4], r[1] - r[3], r[1] - r[5]]
        keep(
            proxy + "_tools",
            c,
            np.column_stack(
                r + contrasts + [contrasts[2] - contrasts[0], contrasts[3] - contrasts[1]]
            ),
            [
                "ESCS_AI_RD",
                "proxy_AI_RD",
                "ESCS_computer_RD",
                "proxy_computer_RD",
                "ESCS_internet_RD",
                "proxy_internet_RD",
                "ESCS_AI_minus_computer",
                "ESCS_AI_minus_internet",
                "proxy_AI_minus_computer",
                "proxy_AI_minus_internet",
                "change_AI_minus_computer",
                "change_AI_minus_internet",
            ],
            len(dc),
        )
    print(proxy, c, len(d), toolok, flush=True)
for family, labels in names.items():
    keys = [k for k in vectors if k.startswith(family + "__")]
    p, U = equal_system([vectors[k] for k in keys])
    se = np.sqrt(np.maximum(U.diagonal(), 0))
    np.save(ROOT / "outputs/qa" / f"SES_{family}_cov.npy", U)
    for j, l in enumerate(labels):
        rows.append(
            dict(
                family=family,
                system="EQUAL_SYSTEM",
                metric=l,
                estimate=p[j],
                se=se[j],
                ci_low=p[j] - 1.959963984540054 * se[j],
                ci_high=p[j] + 1.959963984540054 * se[j],
                p_exploratory=float(2 * norm.sf(abs(p[j] / se[j]))) if se[j] > 0 else np.nan,
                n=sum(sizes[(family, k.split("__")[1])] for k in keys),
                systems=len(keys),
            )
        )
pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/SES_estimates.csv", index=False)
np.savez_compressed(ROOT / "outputs/qa/SES_replicates.npz", **vectors)
(ROOT / "outputs/qa/SES_manifest.json").write_text(
    json.dumps(
        {
            "names": names,
            "protocol": protocol,
            "limits": "HISEI/HISCED are partial SES proxies; ESCS components, no objective criterion; paired differences compare socioeconomic definitions not an intervention",
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(
    pd.DataFrame(rows)
    .query(
        "system == 'EQUAL_SYSTEM' and (family.str.endswith('main') or family.str.endswith('tools'))",
        engine="python",
    )
    .to_string(index=False)
)
