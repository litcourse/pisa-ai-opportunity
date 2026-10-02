# PISA AI opportunity analysis

Python code for survey-weighted analysis of reported AI availability at school
and outside school in the PISA 2025 public-use student dataset.

The analyses cover joint availability states, socioeconomic risk differences,
paired comparisons with computers and internet access, within- and between-school
decomposition, selection and pattern-mixture sensitivity, cross-item response
patterns, and alternative parental socioeconomic measures. Inference uses the
80 replicate weights with BRR-Fay = 0.5 and equal weighting of the fixed system set.
Reported availability is a questionnaire classification. Supplementary analyses
are exploratory; these are observational estimates.

## Run

Use Python 3.13 and install the pinned dependencies:

```sh
python -m pip install -r requirements.txt
python run.py test
```

Download the student archive from the
[official database](https://www.oecd.org/en/data/datasets/pisa-2025-database.html)
or the [recorded archive](https://webfs.oecd.org/pisa2022/2025/CY09_MS_STU_PUF.zip).
Extract `CY09_MS_STU_PUF.sav` with a Deflate64-capable tool such as 7-Zip or
Windows Expand-Archive. Keep the original file unchanged. Then run:

```sh
python run.py all --sav /path/to/CY09_MS_STU_PUF.sav
```

Preparation checks the recorded SAV size and CRC32, variable labels, eligible
systems, and socioeconomic cutpoints. The reference complete-response domain
contains 243,364 students in 11,075 schools across 40 systems. The paired-tool
comparison uses 37 systems. The raw archive's SHA-256 and data source are recorded
in `config/input.json`; data release changes can cause verification to fail.

Intermediate records stay in `data_processed/`. Generated aggregate estimates,
diagnostics, and five figures are written to `outputs/`. Neither directory is
tracked. The final step checks numerical estimates against compact reference
fixtures; success is recorded in `outputs/qa/reproduction.json`.

You can also run stages separately:

```sh
python run.py prepare --sav /path/to/CY09_MS_STU_PUF.sav
python run.py analyze
python run.py figures
python run.py verify
```

The SAV is about 2.2 GB. Allow several GB of working memory and disk space.
The full pipeline refits all survey replicates and takes longer than the unit tests.

## Files

- `run.py`: pipeline entry point.
- `scripts/prepare.py`: source verification and local input preparation.
- `scripts/survey.py`, `adjustment.py`, `target.py`: survey estimators and model helpers.
- Other `scripts/` modules: named analyses and figure generation.
- `config/`: fixed analysis settings, selected official variable definitions,
  data fingerprints, and numerical check fixtures.
- `tests/`: independent estimator, optimizer, and sensitivity edge-case checks.

Obtain microdata directly from its publisher and follow the publisher's data
terms. This repository contains code and configuration only.
