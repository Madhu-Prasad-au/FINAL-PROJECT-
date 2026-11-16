# Voyage · Travel Intelligence

Streamlit app for flight fare estimates, the saved name classifier, and hotel recommendations from the supplied travel history.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first gender classification loads `sentence-transformers/all-MiniLM-L6-v2`; this downloads the embedding model and can take a few minutes.

## Deploy to Render

Push this project to a Git repository, create a new Render Blueprint from that repository, and deploy the detected `render.yaml` service. The Docker image includes the CSV datasets and trained `.joblib`/`.pkl` model artifacts.

The classifier is an experimental prediction based on first-name patterns in the training data. Flight estimates and hotel suggestions reflect historical records and are not live quotes or availability.
