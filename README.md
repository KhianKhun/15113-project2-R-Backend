# Project 2 Data Studio Backend

FastAPI + pandas backend for the Data Studio project.

This backend is UI-driven and whitelist-only. It never executes user-provided code.

## Deployed URL

- Backend (Render): `https://one5113-project2-r-backend.onrender.com`

## Features

- Upload CSV and return `dataset_id`
- Preview/schema/summary responses for current dataset state
- Download current dataset as CSV
- Whitelist transforms:
  - `drop_na_rows`
  - `filter_rows` with operators:
    - factor/string: `==`, `!=`, `contains`, `startswith`, `endswith`
    - numeric: `==`, `!=`, `<`, `<=`, `>`, `>=`, `exp`, `log`, `^`, `+`
  - `create_new_column` support for math operators
- Immutable-like versioning:
  - changed result -> new `dataset_id`
  - unchanged result -> returns same `dataset_id`
- Plot rendering endpoints (histogram/scatter/boxplot/line/bar)
- Regression module:
  - linear / polynomial / kernel / knn / spline / additive / logistic
  - model storage in memory
  - fitted curve rendering
  - prediction endpoint
  - linear model p-value reporting

## API Endpoints

- `GET /api/health`
- `POST /api/datasets/upload`
- `GET /api/datasets/{dataset_id}/preview?limit=50`
- `POST /api/datasets/{dataset_id}/transform`
- `GET /api/datasets/{dataset_id}/download`
- `POST /api/datasets/{dataset_id}/plots/render`
- `POST /api/datasets/{dataset_id}/regressions/fit`
- `GET /api/regressions/{model_id}/curve`
- `POST /api/regressions/{model_id}/predict`

## Transform Request Example

```json
{
  "operations": [
    {
      "op": "drop_na_rows",
      "args": { "subset": ["colA"] }
    },
    {
      "op": "filter_rows",
      "args": {
        "clauses": [{ "col": "city", "op": "contains", "value": "burgh" }]
      }
    },
    {
      "op": "filter_rows",
      "args": {
        "clauses": [{ "col": "income", "op": "log", "value": null }],
        "create_new_column": true
      }
    }
  ]
}
```

## Local Run

1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy env file:

```bash
cp .env.example .env
```

4. Start backend:

```bash
uvicorn app.main:app --reload --port 8000
```

## Environment Variables

- `ALLOWED_ORIGINS` (comma-separated CORS origins)
- `MAX_FILE_BYTES` (upload limit)
- `MAX_DATASET_ROWS`
- `MAX_DATASET_COLS`

Example:

```env
ALLOWED_ORIGINS=https://khiankhun.github.io,http://localhost:5173
MAX_FILE_BYTES=5242880
MAX_DATASET_ROWS=200000
MAX_DATASET_COLS=200
```

## Safety and Limits

- No arbitrary R/Python code execution.
- Strict whitelist transform and plotting/regression handlers.
- Upload size and dataframe shape limits enforced.
- Invalid operations return readable `400` responses.

## Troubleshooting

- GitHub Pages frontend shows `Failed to fetch`:
  - verify backend is up: `/api/health`
  - verify Render `ALLOWED_ORIGINS` includes `https://khiankhun.github.io`
  - redeploy backend after env updates
- Transform seems to do nothing:
  - unchanged outputs intentionally return the same `dataset_id`
- Data/model loss after restart:
  - dataset and regression model storage are in-memory (non-persistent)

## Folder Structure

```text
.
|-- app/
|   |-- main.py
|   |-- models.py
|   |-- storage.py
|   |-- transforms.py
|   |-- summary.py
|   |-- routers/
|   |   |-- datasets.py
|   |   |-- transform.py
|   |   |-- plots.py
|   |   `-- regressions.py
|   |-- plotting/
|   `-- regression/
|-- requirements.txt
|-- .env.example
|-- prompt_log.md
`-- README.md
```
