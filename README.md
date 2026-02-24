# Project 2 Data Studio Backend

FastAPI + pandas backend for the Data Studio MVP.

This backend is UI-driven and whitelist-only. It never executes user-provided code.

## Features (MVP)

- Upload CSV and get `dataset_id`
- Get preview, schema, and summary
- Download current dataset CSV
- Apply whitelist transforms:
  - `drop_na_rows`
  - `filter_rows`
- Immutable dataset versions: each transform returns a new `dataset_id`

## API

- `POST /api/datasets/upload`
- `GET /api/datasets/{dataset_id}/preview?limit=50`
- `POST /api/datasets/{dataset_id}/transform`
- `GET /api/datasets/{dataset_id}/download`

Transform body format:

```json
{
  "operations": [
    { "op": "drop_na_rows", "args": { "subset": ["colA"] } },
    {
      "op": "filter_rows",
      "args": {
        "logic": "AND",
        "clauses": [{ "col": "age", "op": ">=", "value": 18 }]
      }
    }
  ]
}
```

## Local Run

1. Create and activate virtual environment.
2. Install deps:

```bash
pip install -r requirements.txt
```

3. Copy env file:

```bash
cp .env.example .env
```

4. Run:

```bash
uvicorn app.main:app --reload --port 8000
```

## Safety Limits

- No arbitrary user code execution (no R/Python eval/exec).
- Upload file size limit via `MAX_FILE_BYTES`.
- Dataset row/column limits via `MAX_DATASET_ROWS` and `MAX_DATASET_COLS`.
- Input validation returns readable `400` errors for invalid user operations.

## Folder Structure

```text
.
├─ app/
│  ├─ main.py
│  ├─ models.py
│  ├─ storage.py
│  ├─ transforms.py
│  ├─ summary.py
│  └─ routers/
│     ├─ datasets.py
│     └─ transform.py
├─ requirements.txt
├─ .env.example
└─ prompt_log.md
```
