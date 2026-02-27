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
- Implemented whitelist plotting for histogram/scatter/boxplot/line/bar with NA-dropping behavior.
- Added regression module (`app/regression/`) with 7 model types and in-memory model storage.
- Added regression fit endpoint (`POST /api/datasets/{dataset_id}/regressions/fit`) and fitted-curve endpoint (`GET /api/regressions/{model_id}/curve`).
- Added prediction endpoint (`POST /api/regressions/{model_id}/predict`) using stored fitted models.
- Added curve overlay behavior: prediction point + green dashed guides when plot_x is provided.
- Added `line_p_value` metric for linear regression.

## Human vs AI Contribution

- Human: requirements, architecture constraints, feature scope.
- AI: scaffolding, endpoint implementation, transform logic, docs templates.
