"""Streamlit app: estimate the nightly price of a London Airbnb listing."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "price_model.joblib"
CHOICES_PATH = ROOT / "reports" / "choices.json"
METRICS_PATH = ROOT / "reports" / "metrics.json"

st.set_page_config(page_title="London Airbnb Price Estimator", page_icon="🏠")
st.title("🏠 London Airbnb Price Estimator")
st.caption("Estimate a fair nightly price for a London listing from its features.")

if not MODEL_PATH.exists() or not CHOICES_PATH.exists():
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()


@st.cache_resource
def load():
    return joblib.load(MODEL_PATH), json.loads(CHOICES_PATH.read_text())


model, ch = load()
hoods = ch["neighbourhoods"]

c1, c2 = st.columns(2)
with c1:
    hood = st.selectbox("Borough / neighbourhood", sorted(hoods))
    room = st.selectbox("Room type", ch["room_type"])
    prop = st.selectbox("Property type", ch["property_type"])
    superhost = st.selectbox("Superhost", ["f", "t"], format_func=lambda x: "Yes" if x == "t" else "No")
with c2:
    accommodates = st.slider("Guests it accommodates", 1, 16, 2)
    bedrooms = st.slider("Bedrooms", 0, 10, 1)
    beds = st.slider("Beds", 1, 16, 1)
    bathrooms = st.slider("Bathrooms", 0.0, 8.0, 1.0, step=0.5)
    min_nights = st.slider("Minimum nights", 1, 30, 2)
    amenities = st.slider("Number of amenities", 0, 100, 30)

if st.button("Estimate price", type="primary"):
    d = ch["defaults"]
    row = pd.DataFrame([{
        "accommodates": accommodates, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "beds": beds, "minimum_nights": min_nights,
        "availability_365": d["availability_365"], "number_of_reviews": d["number_of_reviews"],
        "review_scores_rating": d["review_scores_rating"],
        "latitude": hoods[hood]["latitude"], "longitude": hoods[hood]["longitude"],
        "amenities_count": amenities, "room_type": room,
        "neighbourhood_cleansed": hood, "property_type": prop, "host_is_superhost": superhost,
    }])
    price = float(np.expm1(model.predict(row)[0]))
    st.metric("Estimated price per night", f"{price:,.0f}")

if METRICS_PATH.exists():
    with st.expander("About the model"):
        m = json.loads(METRICS_PATH.read_text())
        st.write(f"**Model:** {m['best_model']}  |  **Test MAE:** {m['test_mae']:.1f} per night  |  "
                 f"**R²:** {m['test_r2']:.2f}")
        st.caption("Data: Inside Airbnb (insideairbnb.com), licensed CC BY 4.0.")
        for img in ["feature_importance.png", "actual_vs_predicted.png", "price_distribution.png"]:
            if (ROOT / "reports" / img).exists():
                st.image(str(ROOT / "reports" / img))
