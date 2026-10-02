"""Export analysis scientific figures from unchanged aggregate estimates."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)
T = ROOT / "outputs/tables"
A = pd.read_csv(T / "estimates.csv")
S = pd.read_csv(T / "sensitivities.csv")
IP = pd.read_csv(T / "MAR_selection_sensitivity.csv")
F = json.loads((ROOT / "config/analysis.json").read_text())
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)


def export(fig, n):
    for ext in ["png", "pdf", "eps"]:
        fig.savefig(OUT / f"Fig{n}.{ext}", dpi=600)
    plt.close(fig)


data = (
    A.query("family=='main' and system!='EQUAL_SYSTEM'")
    .pivot(index="system", columns="metric", values="estimate")
    .sort_index()
)
fig, ax = plt.subplots(figsize=(6.85, 8.0))
left = np.zeros(len(data))
for key, color, hatch, label in zip(
    ["both", "school_only", "outside_only", "neither"],
    ["#426A8D", "#71B7A1", "#D9B56B", "#AC5968"],
    ["", "///", "...", "xx"],
    ["Both settings", "School only", "Outside school only", "Neither setting"],
):
    values = data[key].to_numpy() * 100
    ax.barh(
        np.arange(len(data)),
        values,
        left=left,
        color=color,
        hatch=hatch,
        edgecolor="white",
        linewidth=0.15,
        label=label,
        height=0.72,
    )
    left += values
ax.set(
    yticks=np.arange(len(data)),
    yticklabels=data.index,
    xlim=(0, 100),
    xlabel="Reported AI availability (%)",
)
ax.invert_yaxis()
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=2, frameon=False)
fig.subplots_adjust(left=0.105, right=0.98, bottom=0.075, top=0.90)
export(fig, 1)

d = A.query("family=='main' and metric=='neither_RD' and system!='EQUAL_SYSTEM'").sort_values(
    "estimate"
)
agg = A.query("family=='main' and metric=='neither_RD' and system=='EQUAL_SYSTEM'")
d = pd.concat([d, agg])
fig, ax = plt.subplots(figsize=(6.85, 8.0))
yy = np.arange(len(d))
ax.errorbar(
    d.estimate * 100,
    yy,
    xerr=np.vstack([(d.estimate - d.ci_low) * 100, (d.ci_high - d.estimate) * 100]),
    fmt="o",
    color="#426A8D",
    markersize=3,
    capsize=1.5,
)
ax.axvline(0, color="gray", linewidth=0.7)
ax.set(
    yticks=yy,
    yticklabels=d.system,
    xlabel="Double unavailability: low SES minus high SES\n(percentage points)",
)
ax.invert_yaxis()
fig.subplots_adjust(left=0.20, right=0.98, bottom=0.10, top=0.98)
export(fig, 2)

j = pd.read_csv(T / "joint_components.csv").query("system=='EQUAL_SYSTEM'").set_index("metric")
grid = pd.read_csv(T / "MNAR_grid.csv").query("system=='EQUAL_SYSTEM'")
fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.85, 6.5), gridspec_kw={"height_ratios": [1, 1.35]})
keys = ["school_margin_contribution", "outside_margin_contribution", "alignment_RD", "joint_RD"]
vals = np.array([j.loc[k, "estimate"] * 100 for k in keys])
lo = np.array([j.loc[k, "ci_low"] * 100 for k in keys])
hi = np.array([j.loc[k, "ci_high"] * 100 for k in keys])
bars = ax.barh(np.arange(4), vals, color=["#B66D51", "#3C8D9A", "#76598A", "#2F3542"])
for bar, hatch in zip(bars, ["/", "\\", "..", ""]):
    bar.set_hatch(hatch)
ax.errorbar(vals, np.arange(4), xerr=[vals - lo, hi - vals], fmt="none", color="black", capsize=3)
ax.axvline(0, color="gray", linewidth=0.7)
ax.set(
    yticks=np.arange(4),
    yticklabels=[
        "School margin",
        "Outside margin",
        "Alignment difference",
        "Joint risk difference",
    ],
    xlabel="Q1 minus Q4 (percentage points)",
    title="A  Components of the paired-report gradient",
)
ax.invert_yaxis()
z = grid.pivot(index="delta_q1", columns="delta_q4", values="estimate") * 100
l = grid.pivot(index="delta_q1", columns="delta_q4", values="ci_low") * 100
h = grid.pivot(index="delta_q1", columns="delta_q4", values="ci_high") * 100
x, y = np.meshgrid(z.columns.to_numpy(), z.index.to_numpy())
m = max(abs(z.to_numpy().min()), abs(z.to_numpy().max()))
im = bx.pcolormesh(
    x,
    y,
    z.to_numpy(),
    cmap="RdBu",
    norm=TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m),
    shading="nearest",
)
bx.contour(x, y, z.to_numpy(), levels=[0], colors="black", linewidths=1.2)
bx.contour(x, y, l.to_numpy(), levels=[0], colors="black", linestyles="--", linewidths=0.8)
bx.contour(x, y, h.to_numpy(), levels=[0], colors="black", linestyles="--", linewidths=0.8)
tipping = pd.read_csv(T / "MNAR_tipping.csv").query("threshold=='point_zero'").iloc[0]
bx.plot(tipping.delta_q4, 0, "o", color="white", markeredgecolor="black", ms=5)
bx.set(
    xlim=(-2, 2),
    ylim=(-2, 2),
    xlabel="Q4 ambiguous-outcome log-odds shift",
    ylabel="Q1 ambiguous-outcome log-odds shift",
    title="B  Broader-target direction under assumed shifts",
)
fig.colorbar(im, ax=bx, label="Conditional risk difference (pp)", pad=0.025)
fig.subplots_adjust(left=0.24, right=0.97, bottom=0.09, top=0.95, hspace=0.54)
export(fig, 3)

order = F["systems"] + ["EQUAL_SYSTEM"]
fig, ax = plt.subplots(figsize=(6.85, 8.0))
for metric, offset, color, marker, label in [
    ("within_contribution", -0.15, "#AC5968", "o", "Within-school contribution"),
    ("between_contribution", 0.15, "#426A8D", "s", "Between-school contribution"),
]:
    d = A.query("family=='decomposition' and metric==@metric").set_index("system").loc[order]
    ax.errorbar(
        d.estimate * 100,
        np.arange(len(d)) + offset,
        xerr=np.vstack([(d.estimate - d.ci_low) * 100, (d.ci_high - d.estimate) * 100]),
        fmt=marker,
        markersize=2.8,
        color=color,
        capsize=1,
        label=label,
    )
ax.axvline(0, color="gray", linewidth=0.7)
ax.set(
    yticks=np.arange(len(order)),
    yticklabels=order,
    xlabel="School availability contribution\n(percentage points per ESCS unit)",
)
ax.invert_yaxis()
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), frameon=False)
fig.subplots_adjust(left=0.20, right=0.98, bottom=0.10, top=0.90)
export(fig, 4)

rows = []
for family, metric, label in [
    ("main", "neither_RD", "Primary complete responses"),
    ("common_sample_sensitivity", "neither_RD", "Same three-tool respondents"),
    ("recomputed_quartiles", "neither_RD", "Replicate-recomputed quartiles"),
]:
    v = A.query("family==@family and metric==@metric and system=='EQUAL_SYSTEM'").iloc[0]
    rows.append((label, v.estimate, v.ci_low, v.ci_high))
for metric, label in [
    ("LPM_adjusted_RD", "Adjusted linear probability"),
    ("regularized_logit_standardized_RD", "Adjusted regularized logit"),
]:
    v = S.query("family=='covariate_adjusted' and metric==@metric and system=='EQUAL_SYSTEM'").iloc[
        0
    ]
    rows.append((label, v.estimate, v.ci_low, v.ci_high))
v = IP.query("system=='EQUAL_SYSTEM'").iloc[0]
rows.append(("Conditional MAR\nselection weighting", v.estimate, v.ci_low, v.ci_high))
fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.85, 5.2), gridspec_kw={"height_ratios": [2, 1]})
v = np.array([r[1:] for r in rows]) * 100
ax.errorbar(
    v[:, 0],
    np.arange(len(rows)),
    xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
    fmt="o",
    color="#426A8D",
    capsize=3,
    markersize=4,
)
ax.set(
    yticks=np.arange(len(rows)),
    yticklabels=[r[0] for r in rows],
    xlabel="Risk difference (percentage points)",
    title="A  Sample and model sensitivity: 95% design intervals",
)
ax.invert_yaxis()
ax.axvline(0, color="gray", linewidth=0.7)
b = S.query("family=='missingness_bounds' and system=='EQUAL_SYSTEM'").set_index("metric")
lb = b.loc["RD_lower_bound", "estimate"] * 100
ub = b.loc["RD_upper_bound", "estimate"] * 100
bx.hlines(0, lb, ub, color="#AC5968", linewidth=4)
bx.scatter([lb, ub], [0, 0], color="#AC5968", s=20)
bx.axvline(0, color="gray", linewidth=0.7)
bx.set(
    yticks=[],
    ylim=(-0.35, 0.35),
    xlabel="Risk difference (percentage points)",
    title="B  Unrestricted identification region; not a confidence interval",
)
bx.text(lb, 0.10, f"{lb:.3f}", ha="center")
bx.text(ub, 0.10, f"{ub:.3f}", ha="center")
fig.subplots_adjust(left=0.34, right=0.97, bottom=0.12, top=0.91, hspace=0.62)
export(fig, 5)
print("Exported five analysis figures in PNG (600 dpi), vector PDF, and EPS.")
