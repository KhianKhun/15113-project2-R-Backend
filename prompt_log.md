# Prompt Log (Backend)

## Tools / Models Used

- Codex (GPT-5 coding agent)

## Session Notes

### 2026-02-24

- Built backend MVP skeleton with FastAPI + pandas.
- Implemented upload/preview/transform endpoints.
- Implemented whitelist transform dispatcher:
  - `drop_na_rows`
  - `filter_rows`
- Added schema inference and NA summary generation.
- Added input validation and readable error responses.
- Added dataset CSV download endpoint (`GET /api/datasets/{dataset_id}/download`).
- Added plotting module (`app/plotting/`) and plot render endpoint (`POST /api/datasets/{dataset_id}/plots/render`).
- Implemented whitelist plotting for (comparison)histogram/scatter/(comparison)boxplot/line/bar with NA-dropping behavior.
- Added regression module (`app/regression/`) with 7 model types and in-memory model storage.
  - `Linear regression`
  - `Polynomial regression`
  - `Kernel Smoother`
  - `KNN Smoother`
  - `Spline Smoother`
  - `Additive Model`
  - `Logistic Regression`
   (Gpt gives the sample code and required package. I change/write based on that.)
- Added regression fit endpoint (`POST /api/datasets/{dataset_id}/regressions/fit`) and fitted-curve endpoint (`GET /api/regressions/{model_id}/curve`).
- Added prediction endpoint (`POST /api/regressions/{model_id}/predict`) using stored fitted models.
- Added curve overlay behavior: prediction point + green dashed guides when plot_x is provided.
- Added `line_p_value` metric for linear regression.

### 2026-02-26

- Expanded transform/filter behavior:
  - removed `is_in` operator
  - removed `logic` requirement in filter args
  - added math operators `exp`, `log`, `^`, `+`
  - added `create_new_column` option for math transformations

### 2026-02-27

- Updated transform routing behavior:
  - when transformed dataframe is unchanged, backend now returns the same `dataset_id`
  - prevents duplicate history entries on repeated identical operations
- Added stricter numeric validation for math operators (including finite checks).
- Added auto-generated safe column naming for `create_new_column` mode (suffix + collision handling).
- Continued deployment support work for Render + GitHub Pages:
  - CORS expectation via `ALLOWED_ORIGINS`
  - production API base URL integration checks
- Updated backend docs to match latest API and transform behavior.

## Human vs AI Contribution

- Human: requirements, architecture constraints, feature scope, deployment decisions.
- AI: scaffolding, endpoint implementation, transform/regression/plot logic, state/version behavior updates, docs updates.
