from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).parent


@st.cache_data
def load_data():
    flights = pd.read_csv(ROOT / "flights.csv", encoding="latin1")
    hotels = pd.read_csv(ROOT / "hotels.csv", encoding="latin1")
    users = pd.read_csv(ROOT / "users.csv", encoding="latin1")
    return flights, hotels, users


@st.cache_resource
def load_flight_model():
    model = joblib.load(ROOT / "flight_price_model_rf.joblib")
    encoders = {
        "from": joblib.load(ROOT / "le_from.joblib"),
        "to": joblib.load(ROOT / "le_to.joblib"),
        "flightType": joblib.load(ROOT / "le_flightType.joblib"),
        "agency": joblib.load(ROOT / "le_agency.joblib"),
    }
    return model, encoders


@st.cache_resource
def load_gender_model():
    # Delay PyTorch/SentenceTransformer imports until the classifier is requested.
    # This keeps the app's normal startup within Render's free-instance memory.
    from sentence_transformers import SentenceTransformer

    artifact = joblib.load(ROOT / "name_gender_classifier.pkl")
    encoder = artifact["label_encoder"] if "label_encoder" in artifact else joblib.load(ROOT / "label_encoder.pkl")
    classifier = artifact["classifier"] if "classifier" in artifact else joblib.load(ROOT / "classifier.pkl")
    embedder = SentenceTransformer(artifact.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    return classifier, encoder, embedder


def hero():
    st.markdown("""
    <style>
    .stApp {background: #f5f7fb;}
    .block-container {max-width: 1180px; padding-top: 2rem;}
    .hero {padding: 1.5rem 1.8rem; border-radius: 18px; color: white;
      background: linear-gradient(115deg,#102b46,#126a73); margin-bottom: 1.2rem;}
    .hero h1 {margin: 0 0 .25rem 0; font-size: 2.2rem;}
    .hero p {margin: 0; opacity: .85;}
    div[data-testid="stMetric"] {background:white;padding:1rem;border-radius:14px;border:1px solid #e8edf3;}
    </style>
    <div class="hero"><h1>Voyage</h1><p>Travel intelligence · flight estimates · personalized stays</p></div>
    """, unsafe_allow_html=True)


st.set_page_config(page_title="Voyage · Travel Intelligence", page_icon="✈️", layout="wide")
flights, hotels, users = load_data()
hero()

page = st.sidebar.radio("Explore", ["Overview", "Flight price", "Name classifier", "Hotel finder"], label_visibility="visible")
st.sidebar.caption("Models trained from the included Voyage travel datasets.")

if page == "Overview":
    st.subheader("Your travel data at a glance")
    a, b, c, d = st.columns(4)
    a.metric("Flight records", f"{len(flights):,}")
    b.metric("Hotel stays", f"{len(hotels):,}")
    c.metric("Travelers", f"{users['code'].nunique():,}")
    d.metric("Average flight fare", f"${flights['price'].mean():,.0f}")
    left, right = st.columns(2)
    with left:
        route = flights.assign(route=flights["from"] + " → " + flights["to"]).groupby("route").agg(bookings=("price", "size"), fare=("price", "mean")).nlargest(10, "bookings").reset_index()
        st.plotly_chart(px.bar(route, x="route", y="bookings", color="fare", color_continuous_scale="Tealgrn", title="Most traveled routes", labels={"route":"Route", "bookings":"Bookings", "fare":"Average fare"}), use_container_width=True)
    with right:
        by_type = flights.groupby("flightType", as_index=False)["price"].mean().sort_values("price", ascending=False)
        st.plotly_chart(px.bar(by_type, x="flightType", y="price", color="flightType", title="Average fare by cabin", labels={"flightType":"Cabin", "price":"Average fare"}), use_container_width=True, show_plotly_logo=False)
    st.caption("Prices and model outputs reflect the historical data supplied with this project.")

elif page == "Flight price":
    st.subheader("Estimate a flight fare")
    st.write("Choose a route and trip details to get a model estimate.")
    model, encoders = load_flight_model()
    with st.form("flight_form"):
        col1, col2, col3 = st.columns(3)
        origin = col1.selectbox("Origin", sorted(encoders["from"].classes_))
        destination = col2.selectbox("Destination", sorted(encoders["to"].classes_))
        cabin = col3.selectbox("Cabin", sorted(encoders["flightType"].classes_))
        col4, col5, col6 = st.columns(3)
        agency = col4.selectbox("Agency", sorted(encoders["agency"].classes_))
        duration = col5.number_input("Flight time (hours)", min_value=0.1, max_value=40.0, value=3.0, step=0.1)
        distance = col6.number_input("Distance (km)", min_value=1.0, max_value=20000.0, value=1000.0, step=50.0)
        submitted = st.form_submit_button("Estimate fare", type="primary")
    if submitted:
        values = [[encoders["from"].transform([origin])[0], encoders["to"].transform([destination])[0], encoders["flightType"].transform([cabin])[0], encoders["agency"].transform([agency])[0], duration, distance]]
        model_input = pd.DataFrame(values, columns=["from_num", "to_num", "flightType_num", "agency_num", "time", "distance"])
        fare = float(model.predict(model_input)[0])
        st.success(f"Estimated fare: **${fare:,.2f}**")
        st.caption("Estimate from the trained random forest. Actual fares can vary.")

elif page == "Name classifier":
    st.subheader("Name based gender classification")
    st.info("This experimental model infers a dataset label from a first name. It may be inaccurate and does not determine a person's identity.")
    full_name = st.text_input("Enter a name", placeholder="e.g. Alex Morgan")
    if st.button("Classify name", type="primary"):
        first_name = full_name.strip().split()[0] if full_name.strip() else ""
        if not first_name:
            st.warning("Enter a name to continue.")
        else:
            with st.spinner("Preparing name features…"):
                classifier, encoder, embedder = load_gender_model()
                vector = embedder.encode([first_name], show_progress_bar=False)
                prediction = classifier.predict(vector)
                label = encoder.inverse_transform(prediction)[0]
            st.success(f"Model prediction: **{label}**")

elif page == "Hotel finder":
    st.subheader("Find a stay that fits")
    st.write("Recommendations use historical hotel prices and stay lengths for each destination.")
    places = sorted(hotels["place"].dropna().unique().tolist())
    with st.form("hotel_form"):
        col1, col2, col3 = st.columns(3)
        city = col1.selectbox("Destination", places)
        budget = col2.slider("Nightly budget", 0, int(max(1000, hotels["price"].quantile(.98))), (100, 500), step=25)
        nights = col3.slider("Length of stay", 1, 14, (2, 5))
        recommend = st.form_submit_button("Show stays", type="primary")
    if recommend:
        stays = hotels[(hotels["place"] == city) & hotels["price"].between(*budget) & hotels["days"].between(*nights)].copy()
        if stays.empty:
            st.warning("No historical stays match all filters. Try widening the budget or stay length.")
        else:
            summary = stays.groupby("name", as_index=False).agg(nightly_price=("price", "median"), typical_nights=("days", "median"), observed_stays=("travelCode", "count"))
            summary = summary.sort_values(["nightly_price", "observed_stays"], ascending=[True, False]).head(10)
            summary["nightly_price"] = summary["nightly_price"].map(lambda x: f"${x:,.2f}")
            summary["typical_nights"] = summary["typical_nights"].map(lambda x: f"{x:g}")
            st.dataframe(summary.rename(columns={"name":"Hotel", "nightly_price":"Median nightly price", "typical_nights":"Typical nights", "observed_stays":"Historical stays"}), hide_index=True, use_container_width=True)
            chart = stays.groupby("name", as_index=False)["price"].median().sort_values("price")
            st.plotly_chart(px.bar(chart, x="name", y="price", color="price", color_continuous_scale="Tealgrn", title=f"Historical nightly prices · {city}", labels={"name":"Hotel", "price":"Median nightly price"}), use_container_width=True)
