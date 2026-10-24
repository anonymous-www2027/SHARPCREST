# Web-scale Dataset Preparation

For the included Wiki and Quito-Hour 100K/200K/400K/800K cohorts, use the [main README](../README.md#quick-start). No conversion or TSF recomputation is needed for those forecasting runs.

This guide covers preparing a new full-population dataset: raw data → common Parquet → TSF analysis → nested whole-series cohorts. Run the commands from the repository root.

## Directory layout

- `data/general/`: the six included General CSVs and `checksums.json`.
- `data/web/`: the included Wiki and Quito-Hour scale-specific Parquet files.
- `data/web/processed/`: optional full-population common Parquet inputs, generated separately.
- `src/preprocessing/`: conversion, schema validation, TSF analysis and nested selection code.

`reproduce.py --data_root` accepts a directory with the same `general/` and `web/` layout. Use `--canonical_root` for a common-Parquet or packaged Web root directly. The training launcher’s `--data-dir` accepts a CSV directory for General datasets or a dataset-parent directory for Web datasets.

## Installation

Use a separate Python 3.10+ environment for population-level TSF preprocessing:

```bash
pip install -r src/preprocessing/requirements.txt
```

Forecasting uses the environment described in the main README.

## Common Parquet format and TSF selection

Users are responsible for converting their raw datasets into the common Parquet format below. Raw layouts vary across datasets; [src/preprocessing/quito.py](../src/preprocessing/quito.py) provides a conversion example, and [src/preprocessing/schema.py](../src/preprocessing/schema.py) provides the shared writer and schema.

Full-population preprocessing uses one row per series in each `part-*.parquet`:

| Field | Type / meaning |
|---|---|
| `series_id` | int64; contiguous IDs starting at zero within each source dataset |
| `source_id` | string; original series identifier |
| `start_timestamp`, `frequency_seconds` | timestamp[ns] and int32 sampling interval |
| `length`, `n_channels` | int32 time steps and int16 channel count |
| `values` | list&lt;float32&gt;; time-major flattened values, length × channels |

`series_index.parquet` stores `series_id`, `source_id`, `part_file`, `row_group`, `row_in_group`, `length`, `n_channels`, and `start_timestamp`. `metadata.json` records dataset information and ordered `channel_names`. Quito uses `ind_1` as channel 0. The compact bundled forecasting files use a separate format and cannot replace the full population for TSF analysis.

### Quito-Hour conversion example

Quito-Hour conversion sorts numeric `item_id`s and timestamps deterministically:

```bash
python -m src.preprocessing convert-quito-hour --source /path/to/test_hour.parquet \
  --dataset quitobench_hour --output data/web/processed/quitobench_hour
```

Repeat with `--dataset quito_corpus_hour`, its pretraining source, and its corresponding output directory. Wiki and TailedTS require user-prepared canonical Parquet inputs.

### TSF analysis and published nested cohorts

To construct nested cohorts:

```bash
python -m src.preprocessing prepare --canonical-root data/web/processed \
  --dataset quito_hour --scales 100000 200000 400000 800000 \
  --output outputs/quito-cohorts --workers 8 --require-published
```

TSF uses training observations only. Group quotas follow population proportions with integer rounding; deterministic sampling uses selection seed **0**, independently of model seeds. Larger cohorts preserve smaller cohorts. Scale labels approximate candidate-window counts, not series counts. Selection reference L/H is Quito 576/288, Wiki 576/7, and TailedTS 168/48.

The TSF generation code is included in [src/preprocessing/population.py](../src/preprocessing/population.py), using the Quito formulas in [src/preprocessing/_quito_tsf.py](../src/preprocessing/_quito_tsf.py). `prepare` computes and saves `population/population_tsf.parquet`; a precomputed table is optional.

The TSF code handles missing values (`NaN`) internally when computing the metrics. To use a different imputation method, fill missing values when creating the common Parquet files.

`--tsf-table` imports a verified population table. `--require-published` checks cohort IDs and raw-data hashes. Override settings with `--protocol`; supported forecasting datasets accept generated manifests via `reproduce.py --cohort_file`.

### Custom datasets

Preprocessing also accepts **custom dataset names**. Put the common-format files under `<canonical-root>/<dataset>/`; schema, index pointers, series identities and values are validated before TSF calculation. Supply reference input/forecast lengths (`L/H`), seasonal period, and exclusive train/validation/test end positions, measured in observations from each series start:

```bash
python -m src.preprocessing prepare --canonical-root /path/to/canonical \
  --dataset my_dataset --L 168 --H 48 --period 24 \
  --train-end 10000 --valid-end 11000 --test-end 12000 \
  --scales 100000 200000 400000 800000 --output outputs/my-cohorts
```

All series must have `test-end` observations and share these relative split boundaries. Multi-channel inputs use `--channel` (default 0). Provide enough series for the requested scales and explicitly impute missing forecasting values upstream. TSF uses only the train segment; `L/H` determine candidate counts, not TSF scores. Each series contributes `train-end - L - H + 1` candidates. Custom cohorts choose the nearest whole-series count, including the initial 100K anchor; `--anchor-series` overrides that count. Published presets retain their original anchor counts. `--selection-seed` defaults to 0; explicit CLI settings override `--protocol` JSON. Outputs include the population TSF table, nested cohort manifests, selected IDs, quotas and hashes. Custom-name support here covers preprocessing; `reproduce.py` still requires a supported dataset configuration.

## External forecasting inputs

For external inputs, `--data_root /path/to/data` expects:

```text
data/
  general/ETT/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
  general/experiment/{Exchange,Weather}_wide.csv
  web/processed/<source_dataset>/
    metadata.json
    series_index.parquet
    part-*.parquet
```

General CSVs contain `date` and ordered feature columns (ETT: 7, Exchange: 8, Weather: 21). Set `--target` if the target column is not `OT`.
