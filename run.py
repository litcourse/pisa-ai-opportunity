"""Run preparation, analyses, figures, tests, and numerical reference checks."""

from pathlib import Path
import argparse
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version

ROOT = Path(__file__).resolve().parent
STAGES = [
    "estimate.py",
    "sensitivity.py",
    "selection.py",
    "gender.py",
    "measurement.py",
    "mnar.py",
    "parental_ses.py",
    "joint.py",
]


def execute(arguments):
    environment = os.environ.copy()
    environment.update(
        PYTHONIOENCODING="utf-8",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, env=environment, check=True)


def verify():
    import numpy as np
    import pandas as pd

    references = json.loads((ROOT / "config/reference_checks.json").read_text())
    count = 0
    for check in references:
        table = pd.read_csv(ROOT / "outputs/tables" / check["file"])
        mask = np.ones(len(table), dtype=bool)
        for key, expected in check["where"].items():
            mask &= (
                np.isclose(table[key].to_numpy(float), float(expected))
                if pd.api.types.is_numeric_dtype(table[key])
                else table[key].eq(expected).to_numpy()
            )
        selected = table.loc[mask]
        if len(selected) != 1:
            raise ValueError(f'Expected exactly one reference row in {check["file"]}.')
        for key, expected in check["values"].items():
            np.testing.assert_allclose(
                float(selected.iloc[0][key]),
                expected,
                rtol=1e-7,
                atol=1e-9,
                err_msg=f'{check["file"]}: {key}',
            )
            count += 1
    main = pd.read_csv(ROOT / "outputs/tables/estimates.csv")
    states = main.query(
        "family == 'main' and metric in ['both','school_only','outside_only','neither']"
    )
    np.testing.assert_allclose(states.groupby("system").estimate.sum(), 1, atol=1e-12)
    report = {
        "status": "passed",
        "reference_rows": len(references),
        "numeric_checks": count,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ["numpy", "pandas", "scipy", "pyreadstat", "pyarrow", "matplotlib"]
        },
    }
    (ROOT / "outputs/qa/reproduction.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["all", "prepare", "analyze", "figures", "test", "verify"])
    parser.add_argument("--sav", type=Path, help="Path to the extracted official student SAV.")
    args = parser.parse_args()
    for directory in ["data_processed", "outputs/tables", "outputs/qa", "outputs/figures"]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    if args.stage in ["all", "test"]:
        execute(["-m", "unittest", "discover", "-s", "tests", "-v"])
    if args.stage in ["all", "prepare"]:
        command = [ROOT / "scripts/prepare.py"]
        if args.sav:
            command += ["--sav", args.sav.resolve()]
        execute(command)
    if args.stage in ["all", "analyze"]:
        for stage in STAGES:
            print(f"Running {stage}", flush=True)
            execute([ROOT / "scripts" / stage])
    if args.stage in ["all", "figures"]:
        execute([ROOT / "scripts/figures.py"])
    if args.stage in ["all", "verify"]:
        verify()


if __name__ == "__main__":
    main()
