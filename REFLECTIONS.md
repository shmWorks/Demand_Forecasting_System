# 🧠 Retail-IQ Core Wisdom

<syn id="ARTIFACT-ERR" date="2026-04-20" type="FIX">
OBJ: Safe doc creation.
CAUSE: `IsArtifact:true` on project paths fails.
RULE: Use `IsArtifact:true` ONLY for internal brain paths. Project files = `IsArtifact:false`.
</syn>

<syn id="DEP-SYNC" date="2026-04-20" type="ARCH">
OBJ: Env consistency.
CAUSE: Plan vs requirements drift.
RULE: Sync `requirements.txt` & `pyproject.toml` with `README` stack before execution.
</syn>

<syn id="MODULAR-ML" date="2026-04-20" type="ARCH">
OBJ: Reduce agent load.
CAUSE: Big notebooks poison context.
RULE: Move logic from `.ipynb` to `src/retail_iq/`. Notebooks = drivers only.
</syn>

<syn id="API-SHORTCUT" date="2026-04-20" type="OPT">
OBJ: 100x faster turns.
CAUSE: `view_file` costs.
RULE: Keep `[API_MANIFEST]` in `docs/project_summary.md`. Code from signatures, no source read.
</syn>

<syn id="PATH-CENTRAL" date="2026-04-20" type="ARCH">
OBJ: Cross-platform robust.
CAUSE: Relative paths break on restructure.
RULE: ALWAYS use `config.py` constants (`RAW_DATA_DIR`, etc.) for I/O.
</syn>

<syn id="REPO-CLEAN" date="2026-04-20" type="HYG">
OBJ: Stop repo bloat.
CAUSE: Tracking data/plots = merge conflicts + big history.
RULE: `git rm --cached` big data/outputs. Add `data/`, `outputs/`, `.venv/` to `.gitignore`.

- `MISTAKE/PREFERENCE`: Consolidated redundant README files into a single source of truth.
- `WHY`: Duplicate documentation creates confusion during architectural transitions.
- `NEW RULE`: Merge duplicate project documentation immediately after refactoring.