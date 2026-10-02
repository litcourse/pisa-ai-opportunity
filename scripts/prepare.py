"""Prepare local analysis inputs from the unmodified public-use student SAV."""

from pathlib import Path
import argparse
import json
import shutil
import zlib

import numpy as np
import pandas as pd
import pyreadstat

from survey import groups, kish, quantiles

ROOT = Path(__file__).resolve().parents[1]
EXTRA = ["Option_ICTQ", "IC170Q01JA", "IC171Q01JA", "IC170Q04JA", "IC171Q04JA"]
PARENT = ["HISEI", "HISCED", "PAREDINT", "ST255Q01JA"]
BASE = [
    "CNT",
    "CNTSTUID",
    "CNTSCHID",
    "ESCS",
    "AGE",
    "GRADE",
    "IMMIG",
    "IC170Q10DA",
    "IC171Q10DA",
    "W_FSTUWT",
] + [f"W_FSTURWT{i}" for i in range(1, 81)]
ITEMS = ["CNT", "CNTSTUID", "ST004D01T", "ST436Q16DA"] + [f"ST438Q0{i}DA" for i in range(1, 5)]


def special_missing(series, field, dictionary):
    """Preserve ordinary response categories and remove documented special codes."""
    result = series.copy()
    for value, label in dictionary[field]["values"].items():
        if any(
            word in label.lower()
            for word in [
                "valid skip",
                "system missing",
                "not administered",
                "invalid",
                "no response",
            ]
        ):
            try:
                code = float(value)
            except ValueError:
                continue
            result = result.mask(pd.to_numeric(series, errors="coerce") == code)
    return result


def check_source(path, expected):
    if path.stat().st_size != expected["sav_bytes"]:
        raise ValueError(
            "SAV size differs from the reference input; use the recorded data release."
        )
    crc = 0
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            crc = zlib.crc32(block, crc)
    if crc != expected["sav_crc32"]:
        raise ValueError("SAV CRC32 differs from the reference archive member.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sav", type=Path, default=ROOT / "data/raw/student.sav")
    args = parser.parse_args()
    for directory in [
        "data/raw",
        "data_processed",
        "outputs/tables",
        "outputs/qa",
        "outputs/figures",
    ]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    dictionary = json.loads((ROOT / "config/variables.json").read_text(encoding="utf-8"))
    settings = json.loads((ROOT / "config/analysis.json").read_text())
    expected = json.loads((ROOT / "config/input.json").read_text())
    path = args.sav.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Extract the official student archive and supply --sav /path/to/student.sav."
        )
    check_source(path, expected)
    columns = list(dict.fromkeys(BASE + EXTRA + ITEMS + PARENT))
    raw, meta = pyreadstat.read_sav(
        str(path), usecols=columns, user_missing=True, apply_value_formats=False
    )
    if raw.duplicated(["CNT", "CNTSTUID"]).any():
        raise ValueError("Duplicate composite student identifiers.")
    for name in ["IC170Q10DA", "IC171Q10DA"] + ITEMS[3:] + PARENT:
        if meta.column_names_to_labels[name] != dictionary[name]["label"]:
            raise ValueError(f"Unexpected variable label: {name}")
    clean = raw.copy()
    for name in BASE + [i for i in ITEMS[2:] if i not in BASE]:
        clean[name] = special_missing(raw[name], name, dictionary)
        for rule in meta.missing_ranges.get(name, []):
            if pd.api.types.is_numeric_dtype(raw[name]):
                masked = raw[name].between(float(rule["lo"]), float(rule["hi"]))
            else:
                masked = raw[name].astype(str).between(str(rule["lo"]), str(rule["hi"]))
            clean[name] = clean[name].mask(masked)
    clean["ST438Q04DA"] = clean.ST438Q04DA.where(clean.ST438Q04DA.isin(range(1, 6)))
    clean["ST004D01T"] = clean.ST004D01T.where(clean.ST004D01T.isin([1, 2]))
    clean["IMMIG"] = clean.IMMIG.where(clean.IMMIG.isin([1, 2, 3]))
    for name in EXTRA + PARENT:
        clean[name] = special_missing(raw[name], name, dictionary)
    clean[ITEMS].to_parquet(
        ROOT / "data_processed/student_items.parquet", index=False, compression="zstd"
    )
    parent = clean[["CNT", "CNTSTUID"] + PARENT]
    parent.to_parquet(
        ROOT / "data_processed/parental_background.parquet", index=False, compression="zstd"
    )
    audit = {}
    for name in PARENT:
        audit[name] = {
            "SAVlabel": meta.column_names_to_labels[name],
            "codebook": dictionary[name],
            "special_codes": [
                int(k)
                for k, v in dictionary[name]["values"].items()
                if any(
                    w in v.lower()
                    for w in ["skip", "missing", "not administered", "invalid", "no response"]
                )
            ],
            "valid_n": int(parent[name].notna().sum()),
            "min": float(parent[name].min()),
            "max": float(parent[name].max()),
        }
    (ROOT / "outputs/qa/parental_field_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame = clean[BASE + EXTRA].copy()
    weights = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]
    W = frame[weights].to_numpy(float)
    weight_ok = np.isfinite(W).all(axis=1) & (W[:, 0] > 0) & (W[:, 1:] >= 0).all(axis=1)
    pair = frame.IC170Q10DA.between(1, 6) & frame.IC171Q10DA.between(1, 6)
    frame["support_sample"] = (
        pair
        & frame.ESCS.notna()
        & frame.CNTSCHID.notna()
        & frame.CNTSTUID.notna()
        & weight_ok
        & frame.Option_ICTQ.eq(1)
    )
    frame["availability_state"] = np.select(
        [
            frame.IC170Q10DA.eq(6) & frame.IC171Q10DA.eq(6),
            frame.IC170Q10DA.ne(6) & frame.IC171Q10DA.eq(6),
            frame.IC170Q10DA.eq(6) & frame.IC171Q10DA.ne(6),
        ],
        ["neither", "school_only", "outside_only"],
        default="both",
    )
    frame.loc[~pair, "availability_state"] = None
    supported = []
    for system, group in frame.groupby("CNT", sort=True):
        domain = group[group.support_sample]
        good = (
            len(domain) >= settings["minimum_students"]
            and domain.CNTSCHID.nunique() >= settings["minimum_schools"]
        )
        for state in ["both", "school_only", "outside_only", "neither"]:
            cell = domain[domain.availability_state.eq(state)]
            good &= (
                cell.CNTSCHID.nunique() >= settings["minimum_state_schools"]
                and kish(cell.W_FSTUWT.to_numpy()) >= settings["minimum_state_kish"]
            )
        if good:
            supported.append(system)
    if supported != settings["systems"]:
        raise ValueError("Eligible systems differ from the fixed reference set.")
    for system in supported:
        domain = frame[frame.CNT.eq(system) & frame.support_sample]
        cuts = quantiles(domain.ESCS.to_numpy(), domain.W_FSTUWT.to_numpy())
        np.testing.assert_allclose(cuts, settings["quartile_cuts"][system], rtol=0, atol=1e-10)
        groups(domain.ESCS.to_numpy(), cuts)
    domain = frame[frame.CNT.isin(supported) & frame.support_sample]
    if (
        len(frame) != 755721
        or len(domain) != 243364
        or domain.groupby("CNT").CNTSCHID.nunique().sum() != 11075
    ):
        raise ValueError("Reference row or school counts differ.")
    frame.to_parquet(ROOT / "data_processed/availability.parquet", index=False, compression="zstd")
    # The analysis stages read the raw metadata only; a link avoids copying a large SAV.
    destination = ROOT / "data/raw/student.sav"
    if path != destination.resolve() and not destination.exists():
        try:
            destination.hardlink_to(path)
        except OSError:
            try:
                destination.symlink_to(path)
            except OSError:
                shutil.copyfile(path, destination)
    report = {
        "formal_students": len(frame),
        "complete_domain": len(domain),
        "systems": len(supported),
        "schools": int(domain.groupby("CNT").CNTSCHID.nunique().sum()),
        "reference_input_verified": True,
        "quartile_cuts_verified": True,
    }
    (ROOT / "outputs/qa/preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
