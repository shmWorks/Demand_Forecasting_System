---
name: Stabilize Forecasting Pipeline
overview: "Docs-aligned staged remediation: fix critical bugs/stability first, then optimize performance and refactor notebooks into atomic script-backed workflows for local Windows + Kaggle/Colab."
todos:
  - id: audit-docs-code
    content: Audit docs vs code. Build severity-ranked findings register
    status: pending
  - id: fix-critical-stability
    content: Fix highest-severity bugs: crashes, leakage, invalid evaluation
    status: pending
  - id: optimize-resources
    content: Apply memory/CPU/GPU optimizations with low behavior risk
    status: pending
  - id: refactor-notebooks-atomic
    content: Split selected notebooks into atomic script-backed cells
    status: pending
  - id: harden-kaggle-colab
    content: Align dependencies + runtime assumptions for local + Kaggle/Colab
    status: pending
  - id: verify-and-report
    content: Run tests/smoke checks. Publish final remediation report
    status: pending
isProject: false
---

# Stabilize Retail Forecasting and Notebook Portability

## Scope and Outcomes
- Audit code vs docs: [docs/Smart_Retail_Demand_Forecasting_Roadmap.pdf](c:/Users/mypc/Downloads/Retail_Demand_Forecasting/docs/Smart_Retail_Demand_Forecasting_Roadmap.pdf), [docs/Smart_Retail_Demand_Forecasting_Project_Proposal.pdf](c:/Users/mypc/Downloads/Retail_Demand_Forecasting/docs/Smart_Retail_Demand_Forecasting_Project_Proposal.pdf), [docs/Project_slides.html](c:/Users/mypc/Downloads/Retail_Demand_Forecasting/docs/Project_slides.html).
- Priority: critical bug fixes + run stability first. Then performance/resource optimization.
- Refactor notebooks to atomic single-purpose cells: `baseline_models.ipynb`, `advanced_models.ipynb`, `evaluation.ipynb`, `cannibalization.ipynb`, `eda_executed.ipynb`.
- Make workflow reproducible on local Windows + Kaggle/Colab with script-backed notebooks.

## Phase 1: Baseline Audit and Single Source of Truth Findings
- Build severity-ordered findings register:
  - correctness bugs + leakage risks in [src/retail_iq/features.py](c:/Users/mypc/Downloads/Retail_Demand_Forecasting/src/retail_iq/features.py), [src/retail_iq/preprocessing.py](c:/Users/mypc/Downloads/Retail_Demand_Forecasting/src/retail_iq/preprocessing.py), [notebooks/evaluation.ipynb](c:/Users/mypc/Downloads/Retail_Demand_Forecasting/notebooks/evaluation.ipynb), [notebooks/advanced_models.ipynb](c:/Users/mypc/Downloads/Retail_Demand_Forecasting/notebooks/advanced_models.ipynb).
  - performance/memory inefficiencies (duplication, dtype inflation, heavy `.values` copies, repeated transforms).
  - portability blockers (`requirements.txt` vs `pyproject.toml`, runtime assumptions, pathing).
- Validate findings via quick reproducibility checks (import sanity, minimal train/eval smoke path).

## Phase 2: Critical Bug and Stability Fixes
- Remove/replace catastrophic memory patterns (example: dataset x10 duplication in advanced flow).
- Fix high-risk data correctness paths (evaluation date split validity, lag feature leakage behavior).
- Add/adjust lightweight tests in [tests/test_retail_iq.py](c:/Users/mypc/Downloads/Retail_Demand_Forecasting/tests/test_retail_iq.py) for temporal split + feature generation regressions.
- Ensure notebooks run clean on fresh kernel with deterministic seed path.

## Phase 3: Performance and Resource Optimization
- Standardize low-memory dtypes. Remove unnecessary frame/array copies in reusable paths.
- Cache expensive intermediate features to processed parquet once. Reuse downstream.
- Tighten training loops (early stopping, bounded search defaults) for lower CPU/GPU load while preserving quality targets.
- Keep optimizations cross-platform (local + Kaggle/Colab). No hidden environment behavior.

## Phase 4: Script-Backed Notebook Refactor (Atomic Cells)
- Convert selected notebooks into consistent cell blocks:
  - setup/imports
  - config/constants
  - data loading
  - feature build
  - train
  - evaluate
  - save artifacts
  - optional visualization
- Move duplicated heavy logic to reusable functions/modules under [src/retail_iq/](c:/Users/mypc/Downloads/Retail_Demand_Forecasting/src/retail_iq/).
- Keep notebook cells small, rerunnable, side-effect scoped for partial reruns/debugging.

## Phase 5: Kaggle/Colab Compatibility Hardening
- Reconcile manifests so fast-path libs exist in [requirements.txt](c:/Users/mypc/Downloads/Retail_Demand_Forecasting/requirements.txt) and align with [pyproject.toml](c:/Users/mypc/Downloads/Retail_Demand_Forecasting/pyproject.toml).
- Add bootstrap notes + path-safe logic for notebook execution in hosted runtimes.
- Verify importability and end-to-end notebook execution order on clean kernels.

## Phase 6: Verification and Final Report
- Run targeted tests and smoke runs for training/evaluation + notebook import/execute flow.
- Re-run lints for changed files.
- Produce final findings+fix summary: broken parts, changes, measured impact, remaining risk. This stays single source of truth for remediation pass.