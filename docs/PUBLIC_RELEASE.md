# Public Release File Set

`PUBLIC_FILES.txt` is the exact, sorted allowlist for the RAFT/PIR/kNN-MTS + SHARP/CREST release. It includes runtime dependencies, training and grid entrypoints, configuration and cohort manifests, bundled data, preprocessing, portable tests, and licenses. Files not listed are excluded from the export.

## Create the public directory

From the repository root:

```bash
python scripts/export_public.py --output /path/to/new-public-directory
```

The destination must not exist. The exporter checks every copied file against its source and copies no Git history, experiment results, caches, checkpoints, or research-only campaigns. It neither deletes source files nor stages, commits, or pushes anything. Publish the exported directory as the public repository; committing in the original research worktree alone does not exclude already tracked research files.

## Layout

```text
src/        # Forecasting, retrieval, loaders, reproduction and preprocessing
  models/  layers/  data_provider/  exp/  common/  utils/
  reproduction/  preprocessing/  pir/  knn_mts/  run.py
data/
  general/  # ETT/ and experiment/ CSVs; checksums.json
  web/      # Wiki/Quito packaged cohorts
  README.md
configs/    # Experiment presets and cohort identities
scripts/    # Training and export entrypoints
tests/      # Portable validation
reproduce.py
```

Public Python modules use the `src.*` namespace. Run grids with `python -m src.reproduction.sweep` and preprocessing with `python -m src.preprocessing`. The root `reproduce.py` and `scripts/train_forecasting.py` remain the forecasting entrypoints.

## Included scope

| Files | Purpose |
|---|---|
| `reproduce.py`, `src/run.py`, `scripts/train_forecasting.py`, `src/reproduction/` | Shared arguments, train/evaluate entrypoints, grids, runtime configuration |
| `src/models/`, `src/layers/`, `src/utils/`, `src/data_provider/`, `src/exp/`, `src/common/` | RAFT, exact/HNSW retrieval, CREST, Recall diagnostics, loaders and metrics |
| `src/pir/`, `src/knn_mts/`, optional requirements and licenses | PatchTST PIR and DLinear/raw-Pearson kNN-MTS RQ1 paths |
| `configs/` | Presets, nested whole-series IDs, 100K anchors and reference hashes |
| `data/web/` | Wiki and Quito-Hour: separate 100K, 200K, 400K and 800K Parquet files |
| `data/general/ETT/`, `data/general/experiment/`, `data/general/checksums.json` | Six General datasets and their checksums |
| `src/preprocessing/` preprocessing modules and metadata | Common schema, Quito converter, train-only TSF, nested selection, packaging |
| `configs/legacy_frozen_cohorts.json` | Compatibility data still read by `src/data_provider/frozen_panel.py`; no historical campaign Python code is included |
| `tests/`, `requirements-test.txt` | Portable checks, including small four-method training/evaluation tests |
| READMEs, licenses, attribution, `PUBLIC_FILES.txt`, export script | Usage, provenance and release selection |

`src/run.py` remains because the public entrypoint imports its shared argument parser. Keep the complete allowlist rather than selecting only the model and launcher.

## Verify an exported copy

Activate the forecasting environment described in the main README and run from the exported directory:

```bash
pip install -r requirements-test.txt -r requirements-pir.txt -r requirements-knn-mts.txt
python -m pytest tests --ignore=tests/test_dataset_preparation.py -q
python scripts/train_forecasting.py --data-type web --data-dir data/web \
  --dataset extended_wikipedia --scale 200000 --pred-len 7 --method 'S&C' --dry-run
python reproduce.py --dataset quito_hour --scale 200000 --pred_len 288 \
  --canonical_root data/web --prepare_only --output outputs/quito-data-check
```

The portable tests check all bundled Web scales, fixed-anchor targets, method routing, STP/MMR, Recall, loaders, and complete small Web/General training runs. They require no prior Git commit or research output directory. CUDA tests run only where CUDA is available.

For the optional full-population preprocessing tests, use a Python 3.10+ environment with `src/preprocessing/requirements.txt`, the forecasting dependencies, and pytest installed:

```bash
python -m pytest tests/test_dataset_preparation.py -q
```

## Reproducibility boundary

The bundle supports the four methods on the six supplied General datasets and the supplied Wiki/Quito-Hour nested scales. It preserves model/retrieval configuration, cohorts, train-only normalization and fixed-100K evaluation. TailedTS configurations and manifests are included, but its raw forecasting data must be supplied separately. New custom datasets require preprocessing and a compatible forecasting configuration, as described in `data/README.md`.

Portable checks and bounded training establish that the selected files are sufficient to execute the protocol; they do not constitute a fresh full paper-scale training campaign. Identical floating-point scores across environments or multithreaded HNSW builds are not guaranteed. Dependency versions, device availability and thread settings must be recorded for comparisons.

## Layout validation (2026-10-24)

The reorganized export contains 111 files. Its core modules use the `src.*` namespace; no legacy top-level module directory or research campaign is required.

- 313 forecasting/runtime tests and 20 preprocessing tests passed; 20 CUDA-only cases were skipped in CPU validation.
- Eight small Web/General × four-method training/evaluation runs produced exactly the same metrics and prediction arrays before and after relocation.
- The executable ASTs of 27 model/retrieval/training/common modules are unchanged after normalizing imports.
- The 14 bundled CSV/Parquet files and 11 cohort/checksum manifests are byte-identical after relocation.
- All 24 Wiki/Quito scale–horizon data checks and 24 General dataset–horizon checks match the previous layout audit. Fixed-anchor targets remain identical across scales at each horizon.
- Default paths and JSON presets resolve repository data when invoked from another working directory; explicit external roots remain supported.

Validation used an exported directory without Git history or original-worktree imports. Full paper-scale training and full-population TSF recomputation were not repeated. TailedTS still requires separately supplied data.

## Optional model integration validation (2026-10-24)

- The integration audit used an exported set of 166 files, including PIR/PatchTST, kNN-MTS/DLinear, optional dependencies and their license notices.
- All 415 portable forecasting tests passed in both the working tree and an exported directory. This includes the existing RAFT checks and 16 small General/Web four-method training/evaluation paths for the new models.
- Against the original research implementations, 43 PIR and 46 kNN-MTS arrays matched exactly on both CPU and CUDA: selected IDs, scores/distances, controller values, weights where applicable, retrieval predictions and backbone forwards. These comparisons use small fixed inputs and one HNSW build/search thread; they do not assert bitwise equality of independently built multithreaded graphs.
- The six General loaders match the research inputs. Wiki/Quito match the historical normalized values, scalers, query ordering and fixed-test masks at all three horizons. All nine datasets pass both models' data preflight, using externally supplied TailedTS data.
- Existing RAFT computation and bundled data/cohort files are preserved. The two new RQ1 paths fix initial LR to 1e-4 and State/Dispersion/Transition weights to 0.5/0.25/0.25.

The original full benchmark was not retrained during this integration. Tested runtime package versions are recorded in the README. The minimal publication selection omits the integration-only test files and informational model preset JSONs; executable RQ1 defaults remain in the launchers. Run multiprocessing DataLoader tests in an environment that permits worker IPC; restricted sandboxes may block them.
