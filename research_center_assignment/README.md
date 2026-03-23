# Research Center Assignment

This project classifies synthetic UK research centers into quality tiers using an unsupervised K-Means pipeline and exposes predictions through FastAPI.

## What is included

- `EDA_and_Model.ipynb`: notebook with EDA, feature selection, clustering experiments, silhouette analysis, and model export
- `app/`: FastAPI application package
- `data/research_centers.csv`: source dataset
- `models/final_kmeans_pipeline.pkl`: trained pipeline used by the API
- `tests/`: API and regression checks

## Project structure

```text
research_center_assignment/
|
|-- app/
|   |-- __init__.py
|   |-- config.py
|   |-- main.py
|   |-- model.py
|   `-- schemas.py
|-- data/
|   `-- research_centers.csv
|-- models/
|   `-- final_kmeans_pipeline.pkl
|-- tests/
|   |-- conftest.py
|   |-- test_api.py
|   `-- test_legacy_app_checks.py
|-- EDA_and_Model.ipynb
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- README.md
```

## Setup

### Windows PowerShell

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Test the project

Run the full suite:

```bash
pytest -q
```

Run each test file directly:

```bash
pytest tests/test_api.py -q
pytest tests/test_legacy_app_checks.py -q
```

Smoke-test the API after starting Uvicorn:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType 'application/json' `
  -Body '{"internalFacilitiesCount":9,"hospitals_10km":3,"pharmacies_10km":2,"facilityDiversity_10km":0.82,"facilityDensity_10km":0.45}'
```

## API endpoints

- `GET /health`
- `POST /predict`
- `POST /predict/batch`

Example `POST /predict` body:

```json
{
  "internalFacilitiesCount": 9,
  "hospitals_10km": 3,
  "pharmacies_10km": 2,
  "facilityDiversity_10km": 0.82,
  "facilityDensity_10km": 0.45
}
```

## Notes

- The API loads the trained model from `models/final_kmeans_pipeline.pkl`.
- If you retrain the model in the notebook, re-export the updated pipeline to the same path.
- The notebook contains the analysis and business interpretation expected for the assignment.
