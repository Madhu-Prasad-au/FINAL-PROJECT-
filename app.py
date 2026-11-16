from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).parent
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#087e83", "#e4a56d", "#496f84", "#9db7ad", "#7b88aa"]


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
    # Fit a compact character n-gram classifier from the included labeled names.
    # It avoids downloading a transformer model and runs within a small instance.
    _, _, users = load_data()
    names = users["name"].fillna("").astype(str).str.strip().str.split().str[0]
    labels = users["gender"].fillna("").astype(str).str.strip().str.lower()
    valid = names.ne("") & labels.ne("") & labels.ne("none")
    model = Pipeline([
        ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, max_features=12000)),
        ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    model.fit(names[valid], labels[valid])
    return model


def hero():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#172b3a; --muted:#6d7d89; --teal:#087e83; --mint:#dff4ef; --line:#e5eceb; }
    .stApp {background:linear-gradient(180deg,#f4f8f7 0%,#f7f8fa 34rem,#f7f8fa 100%); color:var(--ink);}
    html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
    h1,h2,h3 {font-family:'Manrope',sans-serif; color:var(--ink); letter-spacing:-.035em;}
    .block-container {max-width:1320px; padding:2.2rem 3rem 4rem;}
    [data-testid="stSidebar"] {background:#102c3b; border-right:0;}
    [data-testid="stSidebar"] > div:first-child {padding-top:1.4rem;}
    [data-testid="stSidebar"] * {color:#eaf4f2;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {padding:.55rem .7rem; border-radius:10px;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:#1b4150;}
    .brand {display:flex;align-items:center;gap:.7rem;padding:.3rem .5rem 1.2rem;margin-bottom:1.1rem;border-bottom:1px solid #34505c;}
    .brand-mark {width:40px;height:40px;border-radius:13px;display:grid;place-items:center;background:#b8e6d8;color:#123b42;font-size:21px;}
    .brand-name {font:800 20px Manrope,sans-serif;letter-spacing:-.04em;color:white;}
    .brand-caption {font-size:11px;color:#aac1c5;letter-spacing:.13em;text-transform:uppercase;}
    .hero {position:relative;overflow:hidden;padding:2rem 2.35rem;border-radius:24px;color:white;
      background:radial-gradient(circle at 86% 18%,rgba(115,211,180,.3),transparent 21%),linear-gradient(112deg,#123247 0%,#0e6670 67%,#168986 100%);margin:0 0 2rem;box-shadow:0 18px 44px rgba(16,57,69,.15);}
    .hero:after {content:'✈';position:absolute;right:7%;top:4%;font-size:100px;opacity:.12;transform:rotate(-14deg);}
    .hero-kicker {font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#b9eadd;font-weight:700;margin-bottom:.65rem;}
    .hero h1 {margin:0 0 .4rem;font-size:clamp(2rem,4vw,3rem);line-height:1.08;color:white;}
    .hero p {margin:0;color:#d3e9e6;font-size:15px;max-width:700px;}
    .page-heading {margin:.35rem 0 1.15rem;}
    .page-heading p {margin:.2rem 0;color:var(--muted);font-size:14px;}
    div[data-testid="stMetric"] {background:#fff;padding:1.05rem 1.15rem;border-radius:17px;border:1px solid var(--line);box-shadow:0 5px 18px rgba(22,52,61,.035);}
    [data-testid="stMetricLabel"] {color:#71828b;font-size:12px;font-weight:600;}
    [data-testid="stMetricValue"] {font-family:'Manrope',sans-serif;color:#173b49;font-size:27px;font-weight:800;}
    div[data-testid="stPlotlyChart"] {background:#fff;border:1px solid var(--line);border-radius:18px;padding:12px 12px 2px;box-shadow:0 5px 18px rgba(22,52,61,.035);}
    div[data-testid="stForm"] {background:#fff;border:1px solid var(--line);padding:1.2rem 1.35rem .7rem;border-radius:20px;box-shadow:0 8px 26px rgba(22,52,61,.045);}
    div[data-testid="stTextInput"] input,div[data-testid="stNumberInput"] input {border-radius:10px;}
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {border-radius:10px;}
    div.stButton > button,div[data-testid="stFormSubmitButton"] button {border-radius:11px;padding:.55rem 1.15rem;font-weight:700;box-shadow:0 5px 13px rgba(8,126,131,.16);}
    div[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:15px;overflow:hidden;}
    hr {border-color:var(--line);}
    #MainMenu, footer {visibility:hidden;}
    </style>
    <div class="hero"><div class="hero-kicker">Travel intelligence, thoughtfully designed</div><h1>Make every journey<br>more informed.</h1><p>Explore travel patterns, estimate flight fares, and discover stays shaped by real historical data.</p></div>
    """, unsafe_allow_html=True)


st.set_page_config(page_title="Voyage · Travel Intelligence", page_icon="✈️", layout="wide")
flights, hotels, users = load_data()
hero()

st.sidebar.markdown('<div class="brand"><div class="brand-mark">✦</div><div><div class="brand-name">Voyage</div><div class="brand-caption">Travel intelligence</div></div></div>', unsafe_allow_html=True)
page = st.sidebar.radio("YOUR WORKSPACE", ["Overview", "Flight price", "Name classifier", "Hotel finder"], format_func=lambda item: {"Overview":"◫  Overview", "Flight price":"✈  Flight fares", "Name classifier":"◎  Name insights", "Hotel finder":"⌂  Hotel finder"}[item], label_visibility="visible")
st.sidebar.markdown("<div style='margin:1.5rem .5rem 0;color:#aac1c5;font-size:12px;line-height:1.6'>Insights powered by the Voyage travel datasets.</div>", unsafe_allow_html=True)

if page == "Overview":
    st.markdown("<div class='page-heading'><h2>Travel overview</h2><p>A closer look at routes, fares, stays, and traveler patterns in your dataset.</p></div>", unsafe_allow_html=True)
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
        st.plotly_chart(px.histogram(flights, x="price", nbins=35, color_discrete_sequence=["#168b8b"], title="Flight fare distribution", labels={"price":"Fare"}), use_container_width=True)
    left, right = st.columns(2)
    with left:
        by_type = flights.groupby("flightType", as_index=False)["price"].mean().sort_values("price", ascending=False)
        st.plotly_chart(px.bar(by_type, x="flightType", y="price", color="flightType", title="Average fare by cabin", labels={"flightType":"Cabin", "price":"Average fare"}), use_container_width=True, show_plotly_logo=False)
    with right:
        sample = flights.sample(min(2500, len(flights)), random_state=42)
        st.plotly_chart(px.scatter(sample, x="distance", y="price", color="flightType", opacity=.6, title="Fare vs. flight distance", labels={"distance":"Distance (km)", "price":"Fare", "flightType":"Cabin"}), use_container_width=True)
    left, right = st.columns(2)
    with left:
        agencies = flights.groupby("agency", as_index=False).agg(average_fare=("price", "mean"), flights=("price", "size")).sort_values("average_fare", ascending=False)
        st.plotly_chart(px.bar(agencies, x="agency", y="average_fare", color="flights", color_continuous_scale="Tealgrn", title="Average fare by agency", labels={"agency":"Agency", "average_fare":"Average fare", "flights":"Flight records"}), use_container_width=True)
    with right:
        hotel_places = hotels.groupby("place", as_index=False).agg(median_price=("price", "median"), stays=("price", "size")).nlargest(10, "stays")
        st.plotly_chart(px.bar(hotel_places, x="place", y="median_price", color="stays", color_continuous_scale="Tealgrn", title="Hotel prices in popular destinations", labels={"place":"Destination", "median_price":"Median nightly price", "stays":"Historical stays"}), use_container_width=True)
    left, right = st.columns(2)
    with left:
        gender_counts = users[users["gender"].astype(str).str.lower() != "none"]["gender"].value_counts().rename_axis("gender").reset_index(name="travelers")
        st.plotly_chart(px.pie(gender_counts, names="gender", values="travelers", hole=.48, title="Traveler labels in the dataset", color_discrete_sequence=["#168b8b", "#f0a45d"]), use_container_width=True)
    with right:
        monthly = flights.assign(month=pd.to_datetime(flights["date"], errors="coerce").dt.month).dropna(subset=["month"]).groupby("month", as_index=False)["price"].mean()
        st.plotly_chart(px.line(monthly, x="month", y="price", markers=True, title="Average fare by travel month", labels={"month":"Month", "price":"Average fare"}), use_container_width=True)
    st.caption("Prices and model outputs reflect the historical data supplied with this project.")

elif page == "Flight price":
    st.markdown("<div class='page-heading'><h2>Flight fare estimator</h2><p>Adjust the trip details below to get an estimate from the trained Random Forest model.</p></div>", unsafe_allow_html=True)
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
        submitted = st.form_submit_button("✈  Estimate fare", type="primary", use_container_width=True)
    if submitted:
        values = [[encoders["from"].transform([origin])[0], encoders["to"].transform([destination])[0], encoders["flightType"].transform([cabin])[0], encoders["agency"].transform([agency])[0], duration, distance]]
        model_input = pd.DataFrame(values, columns=["from_num", "to_num", "flightType_num", "agency_num", "time", "distance"])
        fare = float(model.predict(model_input)[0])
        st.markdown(f"<div class='hero' style='margin-top:1.3rem;padding:1.35rem 1.7rem'><div class='hero-kicker'>Predicted fare</div><h1 style='font-size:2.5rem'>${fare:,.2f}</h1><p>Based on the route, cabin, agency, duration, and distance you selected.</p></div>", unsafe_allow_html=True)
        st.caption("Model estimate based on historical fares. Actual prices can vary.")

elif page == "Name classifier":
    st.markdown("<div class='page-heading'><h2>Name insights</h2><p>A lightweight model estimates the label most associated with a first name in this dataset.</p></div>", unsafe_allow_html=True)
    st.info("Experimental prediction only. A name does not determine a person's gender or identity.")
    full_name = st.text_input("First or full name", placeholder="e.g. Alex Morgan")
    if st.button("✦  Analyze name", type="primary"):
        first_name = full_name.strip().split()[0] if full_name.strip() else ""
        if not first_name:
            st.warning("Enter a name to continue.")
        else:
            with st.spinner("Preparing name features…"):
                classifier = load_gender_model()
                label = classifier.predict([first_name])[0]
                confidence = float(classifier.predict_proba([first_name]).max())
            st.markdown(f"<div class='hero' style='margin-top:1.3rem;padding:1.35rem 1.7rem'><div class='hero-kicker'>Most associated dataset label</div><h1 style='font-size:2.5rem'>{label.title()}</h1><p>Model confidence: {confidence:.0%} · This reflects training data and may be wrong.</p></div>", unsafe_allow_html=True)

elif page == "Hotel finder":
    st.markdown("<div class='page-heading'><h2>Find your kind of stay</h2><p>Discover hotels from historical bookings that match your destination, budget, and trip length.</p></div>", unsafe_allow_html=True)
    places = sorted(hotels["place"].dropna().unique().tolist())
    with st.form("hotel_form"):
        col1, col2, col3 = st.columns(3)
        city = col1.selectbox("Destination", places)
        budget = col2.slider("Nightly budget", 0, int(max(1000, hotels["price"].quantile(.98))), (100, 500), step=25)
        nights = col3.slider("Length of stay", 1, 14, (2, 5))
        recommend = st.form_submit_button("⌕  Find matching stays", type="primary", use_container_width=True)
    if recommend:
        stays = hotels[(hotels["place"] == city) & hotels["price"].between(*budget) & hotels["days"].between(*nights)].copy()
        if stays.empty:
            st.warning("No historical stays match all filters. Try widening the budget or stay length.")
        else:
            summary = stays.groupby("name", as_index=False).agg(nightly_price=("price", "median"), typical_nights=("days", "median"), observed_stays=("travelCode", "count"))
            summary = summary.sort_values(["nightly_price", "observed_stays"], ascending=[True, False]).head(10)
            st.markdown(f"<div class='page-heading' style='margin-top:1.5rem'><h3>{len(summary)} stays to explore in {city}</h3><p>{len(stays)} matching historical bookings · nightly budget ${budget[0]:,}–${budget[1]:,}</p></div>", unsafe_allow_html=True)
            summary["nightly_price"] = summary["nightly_price"].map(lambda x: f"${x:,.2f}")
            summary["typical_nights"] = summary["typical_nights"].map(lambda x: f"{x:g}")
            st.dataframe(summary.rename(columns={"name":"Hotel", "nightly_price":"Median nightly price", "typical_nights":"Typical nights", "observed_stays":"Historical stays"}), hide_index=True, use_container_width=True)
            chart = stays.groupby("name", as_index=False)["price"].median().sort_values("price")
            st.plotly_chart(px.bar(chart, x="name", y="price", color="price", color_continuous_scale="Tealgrn", title=f"Historical nightly prices · {city}", labels={"name":"Hotel", "price":"Median nightly price"}), use_container_width=True)
