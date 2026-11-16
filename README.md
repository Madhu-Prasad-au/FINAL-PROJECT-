# Voyage — Travel Intelligence

Voyage is a Streamlit application built from historical flight, hotel, and traveler data. It combines a trained flight-price regression model, an interactive name-based classification demo, hotel recommendations, and an exploratory dashboard.

## Features

- **Travel overview:** Explore historical flight fares, popular routes, cabin types, agencies, hotel prices by destination, traveler labels, and fare trends by month.
- **Flight fare estimates:** Predict a fare from origin, destination, cabin, agency, duration, and distance using the included Random Forest regression model.
- **Name classification:** Estimate the gender label associated with a first name using a lightweight character n-gram classifier trained from `users.csv`. The model is fit on first use and cached for the session; it does not download a large language model.
- **Hotel finder:** Filter historical stays by destination, nightly budget, and number of nights. Results show hotel prices and the number of matching historical stays.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, charts, predictions, and recommendations |
| `flights.csv` | Historical flight records used for flight insights |
| `hotels.csv` | Historical hotel bookings used by the hotel finder |
| `users.csv` | Traveler data and labels used by the name classifier |
| `flight_price_model_rf.joblib` | Trained flight fare regression model |
| `le_from.joblib`, `le_to.joblib`, `le_flightType.joblib`, `le_agency.joblib` | Encoders required by the flight regression model |
| `hotel_features.csv` | Prepared hotel features from the notebook |
| `Voyage-ML.ipynb` | Exploratory analysis and original model development |
| `Dockerfile`, `render.yaml` | Container and Render Blueprint configuration |

## Run locally

Use Python 3.11 or newer.

```bash
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Deploy to Render

The repository includes a `render.yaml` Blueprint and a `Dockerfile`.

1. Push the project to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com/), choose **New + → Blueprint**.
3. Connect GitHub and select this repository.
4. Review the service configuration from `render.yaml` and apply it.
5. Wait for the Docker build and deploy to finish, then open the `onrender.com` service URL.

The Blueprint requests Render's **Free** web-service plan. Free services can spin down after inactivity and may take a short time to respond to the next visit. Free instances have limited CPU and memory, so the app's memory use can affect deployment reliability.

When changing app code or dependencies, commit and push the changes to GitHub. Render will rebuild and redeploy the service from the updated repository.

## Troubleshooting

### PyArrow / NumPy import error

The requirements pin compatible versions of NumPy and PyArrow. If Render reports an old dependency combination, confirm that the latest `requirements.txt` was pushed and trigger a new deploy with **Clear build cache & deploy** in Render.

### App keeps loading or fails to start

Open the service in Render and check **Logs** for the first Python traceback or memory-related restart. Confirm that the latest `app.py`, `requirements.txt`, `Dockerfile`, and `render.yaml` are in the deployed Git commit. On the Free plan, resource limits can cause restarts.

### Hotel results are empty

Hotel suggestions only reflect matching historical records. Try selecting a wider budget range or a different stay length.

## Data and model notes

- Flight fare predictions use the saved regression model and category encoders created during notebook development.
- The name classifier is trained from the labeled names in `users.csv` when first requested. Its prediction is experimental and can be inaccurate; a name does not determine a person's gender or identity.
- Hotel suggestions are based on historical hotel rows. They do not represent current prices, availability, or endorsements.
- Prices and trends shown in the dashboard describe the supplied dataset, not live travel data.
