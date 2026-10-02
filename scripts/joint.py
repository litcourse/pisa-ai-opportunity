"""Post-primary exact accounting of marginal and alignment contributions."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from survey import groups, ratio, equal_system, covariance, inference

ROOT = Path(__file__).resolve().parents[1]
F = json.loads((ROOT / "config/analysis.json").read_text())
METRICS = [
    "u_Q1",
    "u_Q4",
    "v_Q1",
    "v_Q4",
    "d_Q1",
    "d_Q4",
    "alignment_Q1",
    "alignment_Q4",
    "joint_RD",
    "school_margin_contribution",
    "outside_margin_contribution",
    "alignment_RD",
    "margin_reference_RD",
]


def joint_components(u1, u4, v1, v4, d1, d4):
    a1 = d1 - u1 * v1
    a4 = d4 - u4 * v4
    school = (v1 + v4) / 2 * (u1 - u4)
    outside = (u1 + u4) / 2 * (v1 - v4)
    return np.column_stack(
        [u1, u4, v1, v4, d1, d4, a1, a4, d1 - d4, school, outside, a1 - a4, u1 * v1 - u4 * v4]
    )


def main():
    x = pd.read_parquet(ROOT / "data_processed/availability.parquet")
    wc = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]
    all_e = {}
    rows = []
    for system in F["systems"]:
        d = x.loc[x.CNT.eq(system) & x.support_sample]
        q = groups(d.ESCS.to_numpy(), F["quartile_cuts"][system])
        W = d[wc].to_numpy()
        s = d.IC170Q10DA.eq(6).to_numpy()
        o = d.IC171Q10DA.eq(6).to_numpy()
        e = joint_components(
            ratio(s, q == 0, W),
            ratio(s, q == 3, W),
            ratio(o, q == 0, W),
            ratio(o, q == 3, W),
            ratio(s & o, q == 0, W),
            ratio(s & o, q == 3, W),
        )
        assert np.max(np.abs(e[:, 8] - e[:, 9] - e[:, 10] - e[:, 11])) < 1e-12
        all_e[system] = e
        se, lo, hi, p = inference(e[0], covariance(e))
        for k, name in enumerate(METRICS):
            rows.append(
                dict(
                    system=system,
                    metric=name,
                    estimate=e[0, k],
                    se=se[k],
                    ci_low=lo[k],
                    ci_high=hi[k],
                    p=p[k],
                )
            )
    point, cov = equal_system(list(all_e.values()))
    se, lo, hi, p = inference(point, cov)
    for k, name in enumerate(METRICS):
        rows.append(
            dict(
                system="EQUAL_SYSTEM",
                metric=name,
                estimate=point[k],
                se=se[k],
                ci_low=lo[k],
                ci_high=hi[k],
                p=p[k],
            )
        )
    pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/joint_components.csv", index=False)
    np.savez_compressed(ROOT / "outputs/qa/joint_components_replicates.npz", **all_e)
    (ROOT / "outputs/qa/joint_metrics.json").write_text(json.dumps(METRICS, indent=2))
    print(pd.DataFrame(rows).query("system=='EQUAL_SYSTEM'").to_string(index=False))


if __name__ == "__main__":
    main()
