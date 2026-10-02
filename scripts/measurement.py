"""Exploratory supplementary cross-item configurations, never objective validation."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import datetime, json, hashlib
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
ITEMS = {
    "ST436Q16DA": (4, 3, "classroom_AI_information_quality_activity"),
    "ST438Q01DA": (5, 4, "schoolwork_chatbot_summarizing"),
    "ST438Q02DA": (5, 4, "schoolwork_chatbot_research"),
    "ST438Q03DA": (5, 4, "schoolwork_chatbot_drafting"),
    "ST438Q04DA": (5, 4, "schoolwork_chatbot_learning"),
}
STATES = ["both", "school_only", "outside_only", "neither"]
wc = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]
F = json.loads((ROOT / "config/analysis.json").read_text())
D = json.loads((ROOT / "config/variables.json").read_text(encoding="utf-8"))
_, M = pyreadstat.read_sav(str(ROOT / "data/raw/student.sav"), metadataonly=True, user_missing=True)
assert all(i in M.column_names for i in ITEMS)
assert all(M.column_names_to_labels[i] == D[i]["label"] for i in ITEMS)
x = pd.read_parquet(
    ROOT / "data_processed/availability.parquet",
    columns=[
        "CNT",
        "CNTSTUID",
        "CNTSCHID",
        "support_sample",
        "availability_state",
        "IC170Q10DA",
        "IC171Q10DA",
    ]
    + wc,
)
a = pd.read_parquet(
    ROOT / "data_processed/student_items.parquet", columns=["CNT", "CNTSTUID"] + list(ITEMS)
)
x = x.merge(a, on=["CNT", "CNTSTUID"], validate="one_to_one")
support = []
sets = {}
for item, (K, regular, construct) in ITEMS.items():
    sets[item] = []
    for c in F["systems"]:
        d = x[(x.CNT == c) & x.support_sample & x[item].isin(range(1, K + 1))]
        passed = len(d) >= 500 and d.CNTSCHID.nunique() >= 50
        if passed:
            sets[item].append(c)
        for state in ["ALL"] + STATES:
            z = d if state == "ALL" else d[d.availability_state == state]
            w = z.W_FSTUWT.to_numpy()
            k = float(w.sum() ** 2 / (w @ w)) if len(w) else 0
            support.append(
                {
                    "item": item,
                    "system": c,
                    "state": state,
                    "n": len(z),
                    "schools": z.CNTSCHID.nunique(),
                    "kish": k,
                    "overall_analysis_supported": passed,
                    "conditional_precision_flag": bool(
                        state != "ALL" and (k < 30 or z.CNTSCHID.nunique() < 10)
                    ),
                }
            )
pd.DataFrame(support).to_csv(ROOT / "outputs/tables/measurement_support.csv", index=False)
spec = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "stage": "exploratory supplementary analysis after original main results; not confirmatory or preregistered",
    "parent_systems": F["systems"],
    "item_analysis_systems": sets,
    "rules": {
        "overall": "valid item plus original complete paired opportunity domain; >=500 students and50 schools",
        "conditional": "keep every valid state; Kish<30 orschools<10 is precision flag, never discordance-based deletion",
        "zero_denominator": "report nonestimable; never substitute0",
        "inference": "80BRR Fay0.5; independent fixed-system averages; large-sample sampling CIs; no multiplicity-adjusted confirmation claims",
    },
    "items": {
        i: {
            "label_SAV": M.column_names_to_labels[i],
            "codebook": D[i],
            "valid_categories": K,
            "regular_threshold": r,
            "construct": c,
        }
        for i, (K, r, c) in ITEMS.items()
    },
    "time_place_notes": {
        "ST436Q16DA": "same schoolyear and classroom lessons, but assessing AI output need not mean direct tool access",
        "ST438": "chatbot schoolwork purposes, no explicit same schoolyear or physicalsetting; narrower chatbot tool category",
        "IC": "AI tools at school/outside thisschoolyear, including image tools; selfreport not objectiveinventory",
    },
    "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(ROOT / "outputs/qa/measurement_spec.json").write_text(
    json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
)
# All sample/support decisions above saved before any cross-item weighted association.
rows = []
vectors = {}
labels_by_key = {}
meta_by_key = {}
jointrows = []


def keep(item, system, values, labels, meta):
    assert values.shape == (81, len(labels)) and np.isfinite(values).all()
    key = item + "__" + system
    vectors[key] = values
    labels_by_key[key] = labels
    meta_by_key[key] = meta
    dif = values[1:] - values[0]
    V = dif.T @ dif / 20
    for j, label in enumerate(labels):
        se = np.sqrt(max(V[j, j], 0))
        p = float(2 * norm.sf(abs(values[0, j] / se))) if se > 0 else np.nan
        rows.append(
            dict(
                item=item,
                system=system,
                metric=label,
                estimate=values[0, j],
                se=se,
                ci_low=values[0, j] - 1.959963984540054 * se,
                ci_high=values[0, j] + 1.959963984540054 * se,
                p_exploratory=p,
                systems=1,
                **meta[label],
            )
        )
    return V


for item, (K, regular, construct) in ITEMS.items():
    objects = []
    for c in sets[item]:
        d = x[(x.CNT == c) & x.support_sample & x[item].isin(range(1, K + 1))]
        W = d[wc].to_numpy()
        cat = d[item].to_numpy(int) - 1
        state = pd.Categorical(d.availability_state, categories=STATES).codes
        assert (state >= 0).all()
        index = state * K + cat
        joint = np.array(
            [np.bincount(index, weights=w, minlength=4 * K) / w.sum() for w in W.T]
        ).reshape(81, 4, K)
        vals = [joint.reshape(81, 4 * K)]
        labels = [f"joint_{s}_category{k}" for s in STATES for k in range(1, K + 1)]
        base = {
            "n_domain": len(d),
            "schools_domain": d.CNTSCHID.nunique(),
            "n_valid_common": len(d),
            "precision_flag": False,
        }
        meta = {l: base.copy() for l in labels}
        stateq = {}
        for j, s in enumerate(STATES):
            den = joint[:, j, :].sum(axis=1)
            z = d[state == j]
            w = z.W_FSTUWT.to_numpy()
            kish = float(w.sum() ** 2 / (w @ w)) if len(w) else 0
            if not np.all(den > 0):
                continue
            anyq = joint[:, j, 1:].sum(axis=1) / den
            regq = joint[:, j, regular - 1 :].sum(axis=1) / den
            stateq[s] = [anyq, regq]
            for name, value in [("any", anyq), ("regular", regq)]:
                l = f"{name}_given_{s}"
                vals.append(value[:, None])
                labels.append(l)
                meta[l] = {
                    "n_domain": len(z),
                    "schools_domain": z.CNTSCHID.nunique(),
                    "n_valid_common": len(d),
                    "precision_flag": bool(kish < 30 or z.CNTSCHID.nunique() < 10),
                }
        for kind, mask in [
            ("school_unavailable", d.IC170Q10DA.eq(6).to_numpy()),
            ("school_available", d.IC170Q10DA.ne(6).to_numpy()),
        ]:
            wmask = W[mask]
            den = wmask.sum(axis=0)
            if not np.all(den > 0):
                continue
            z = d[mask]
            w = z.W_FSTUWT.to_numpy()
            kish = float(w.sum() ** 2 / (w @ w))
            for name, cond in [
                ("any", d[item].to_numpy() > 1),
                ("regular", d[item].to_numpy() >= regular),
            ]:
                value = W[mask & cond].sum(axis=0) / den
                l = f"{name}_given_{kind}"
                vals.append(value[:, None])
                labels.append(l)
                meta[l] = {
                    "n_domain": len(z),
                    "schools_domain": z.CNTSCHID.nunique(),
                    "n_valid_common": len(d),
                    "precision_flag": bool(kish < 30 or z.CNTSCHID.nunique() < 10),
                }
        for name, j in [("any", 0), ("regular", 1)]:
            if "both" in stateq and "neither" in stateq:
                l = f"{name}_both_minus_neither"
                vals.append((stateq["both"][j] - stateq["neither"][j])[:, None])
                labels.append(l)
                meta[l] = dict(
                    base,
                    precision_flag=meta[f"{name}_given_both"]["precision_flag"]
                    or meta[f"{name}_given_neither"]["precision_flag"],
                )
            la = f"{name}_given_school_available"
            lu = f"{name}_given_school_unavailable"
            if la in labels and lu in labels:
                # Each scalar is at its aligned index in the concatenated vector.
                matrix = np.column_stack(vals)
                l = f"{name}_school_available_minus_unavailable"
                vals.append((matrix[:, labels.index(la)] - matrix[:, labels.index(lu)])[:, None])
                labels.append(l)
                meta[l] = dict(
                    base, precision_flag=meta[la]["precision_flag"] or meta[lu]["precision_flag"]
                )
        matrix = np.column_stack(vals)
        V = keep(item, c, matrix, labels, meta)
        objects.append((c, matrix, labels, V, meta))
        print(item, c, len(d), flush=True)
    # Different nonestimable conditional rows can have different system sets; covariance uses shared systems only.
    union = list(dict.fromkeys(l for _, _, labs, _, _ in objects for l in labs))
    position = {l: j for j, l in enumerate(union)}
    counts = np.zeros(len(union))
    p = np.zeros(len(union))
    V = np.zeros((len(union), len(union)))
    summeta = {
        l: {
            "n_domain": 0,
            "schools_domain": 0,
            "n_valid_common": 0,
            "precision_flag": False,
            "included_systems": [],
        }
        for l in union
    }
    for c, e, labs, u, mm in objects:
        ix = np.array([position[l] for l in labs])
        counts[ix] += 1
        p[ix] += e[0]
        V[np.ix_(ix, ix)] += u
        for l in labs:
            for n in ["n_domain", "schools_domain", "n_valid_common"]:
                summeta[l][n] += mm[l][n]
            summeta[l]["precision_flag"] |= mm[l]["precision_flag"]
            summeta[l]["included_systems"].append(c)
    p /= counts
    V /= counts[:, None] * counts[None, :]
    np.save(ROOT / "outputs/qa" / f"measurement_{item}_cov.npy", V)
    for j, l in enumerate(union):
        se = np.sqrt(max(V[j, j], 0))
        m = summeta[l].copy()
        m["included_systems"] = ";".join(m["included_systems"])
        rows.append(
            dict(
                item=item,
                system="EQUAL_SYSTEM",
                metric=l,
                estimate=p[j],
                se=se,
                ci_low=p[j] - 1.959963984540054 * se,
                ci_high=p[j] + 1.959963984540054 * se,
                p_exploratory=float(2 * norm.sf(abs(p[j] / se))) if se > 0 else np.nan,
                systems=int(counts[j]),
                **m,
            )
        )
    spec.setdefault("equal_system_metric_order", {})[item] = union
pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/measurement_estimates.csv", index=False)
np.savez_compressed(ROOT / "outputs/qa/measurement_replicates.npz", **vectors)
(ROOT / "outputs/qa/measurement_manifest.json").write_text(
    json.dumps(
        {
            "spec": spec,
            "labels_by_key": labels_by_key,
            "inference": "Exploratory large-sample BRR CIs only; zeros do not establish valid measurement",
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(
    pd.DataFrame(rows)
    .query("system == 'EQUAL_SYSTEM' and not metric.str.startswith('joint')", engine="python")
    .to_string(index=False)
)
