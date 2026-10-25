"""Smartphone Resale Price Predictor - Streamlit web app.

Loads the trained scikit-learn pipeline (smartphone_price_model.joblib) produced by
Used_Smartphone_Price_Prediction_Assignment_3.ipynb and predicts the normalized used
price of a phone or tablet from the 8 features the model was trained on.
"""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "smartphone_price_model.joblib"
METADATA_PATH = APP_DIR / "model_metadata.json"

FIELD_LABELS = {
    "device_brand": "Brand",
    "normalized_new_price": "Price of this model when new (€)",
    "internal_memory": "Internal storage (GB)",
    "screen_size": "Screen size (cm, diagonal)",
    "rear_camera_mp": "Rear camera (MP)",
    "front_camera_mp": "Front camera (MP)",
    "release_year": "Release year",
    "days_used": "Days used",
}
OPTIONAL_FIELDS = []
STORAGE_OPTIONS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

st.set_page_config(page_title="Smartphone Resale Price Predictor", page_icon="📱", layout="centered")


@st.cache_resource(show_spinner="Loading the trained model ...")
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def validate(device, metadata):
    """Return a list of human-readable problems with the input (empty = OK)."""
    problems = []
    for col, info in metadata["numeric_features"].items():
        value = device.get(col)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            if col not in OPTIONAL_FIELDS:
                problems.append(f"**{FIELD_LABELS[col]}** is required.")
            continue
        if value < 0:
            problems.append(f"**{FIELD_LABELS[col]}** cannot be negative.")
    for col, allowed in metadata["categorical_features"].items():
        if device.get(col) not in allowed:
            problems.append(f"**{FIELD_LABELS[col]}** must be one of: {', '.join(allowed)}.")
    return problems


def out_of_range(device, metadata):
    """Numeric inputs outside the range seen in training (prediction is an extrapolation)."""
    notes = []
    for col, info in metadata["numeric_features"].items():
        value = device.get(col)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if value < info["min"] or value > info["max"]:
            notes.append(f"{FIELD_LABELS[col]} = {value:g} (training range {info['min']:g} – {info['max']:g})")
    return notes


def predict(model, metadata, device):
    frame = pd.DataFrame([device])[metadata["feature_order"]]
    value = float(model.predict(frame)[0])
    if not np.isfinite(value):
        raise ValueError("The model returned a non-numeric prediction.")
    return value


def main():
    st.title("📱 Smartphone Resale Price Predictor")
    st.write(
        "Estimate the resale price of a **used or refurbished phone/tablet** from its specifications, "
        "age and the price of the same model when new."
    )

    try:
        model, metadata = load_model()
    except Exception:
        st.error("The prediction model could not be loaded. Please try again later.")
        st.stop()

    cv = metadata["cv"]["summary"][metadata["algorithm"]]
    st.info(
        "**This is a statistical estimate, not a guaranteed market offer.** "
        "Prices are in **euros (€)** at the time the data was collected (2021).",
        icon="ℹ️",
    )

    num = metadata["numeric_features"]
    cats = metadata["categorical_features"]

    with st.form("device_form"):
        st.subheader("Enter the 8 device details")
        c1, c2 = st.columns(2)
        brand = c1.selectbox(FIELD_LABELS["device_brand"], cats["device_brand"],
                             index=cats["device_brand"].index("Samsung") if "Samsung" in cats["device_brand"] else 0,
                             help="Choose 'Others' if your brand is not listed.")
        new_price_eur = c2.number_input(FIELD_LABELS["normalized_new_price"],
                                        min_value=float(math.ceil(math.exp(num["normalized_new_price"]["min"]))),
                                        max_value=float(math.floor(math.exp(num["normalized_new_price"]["max"]))),
                                        value=270.0, step=10.0,
                                        help="Launch / list price of a new unit of the same model, in euros.")
        storage = c1.select_slider(FIELD_LABELS["internal_memory"], options=STORAGE_OPTIONS, value=64)
        screen = c2.number_input(FIELD_LABELS["screen_size"], min_value=float(num["screen_size"]["min"]),
                                 max_value=float(num["screen_size"]["max"]), value=15.34, step=0.1,
                                 help="Diagonal in centimetres (6.1 inch ≈ 15.5 cm).")
        rear = c1.number_input(FIELD_LABELS["rear_camera_mp"], min_value=0.0, max_value=float(num["rear_camera_mp"]["max"]),
                               value=12.0, step=1.0)
        front = c2.number_input(FIELD_LABELS["front_camera_mp"], min_value=0.0, max_value=float(num["front_camera_mp"]["max"]),
                                value=8.0, step=1.0)
        year = c1.slider(FIELD_LABELS["release_year"], int(num["release_year"]["min"]), int(num["release_year"]["max"]), 2019)
        days = c2.slider(FIELD_LABELS["days_used"], int(num["days_used"]["min"]), int(num["days_used"]["max"]), 400)
        submitted = st.form_submit_button("Predict resale price", type="primary", use_container_width=True)

    if not submitted:
        return

    device = {
        "normalized_new_price": math.log(new_price_eur), "internal_memory": float(storage), "rear_camera_mp": rear,
        "front_camera_mp": front, "screen_size": screen, "release_year": int(year), "days_used": int(days),
        "device_brand": brand,
    }

    problems = validate(device, metadata)
    if problems:
        st.error("Please fix the following:\n\n" + "\n".join(f"- {p}" for p in problems))
        return

    try:
        prediction = predict(model, metadata, device)
    except Exception:
        st.error("Sorry, the prediction could not be computed for this input. Please check the values and try again.")
        return

    rmse = cv["mean_rmse"]
    price_eur = math.exp(prediction)
    low_eur, high_eur = math.exp(prediction - rmse), math.exp(prediction + rmse)
    st.subheader("Estimated resale price")
    c1, c2 = st.columns(2)
    c1.metric("Estimated resale price", f"€{price_eur:,.0f}")
    c1.caption(f"Typical range: €{low_eur:,.0f} – €{high_eur:,.0f} (± 1 cross-validation RMSE)")
    c2.metric("Resale value", f"{price_eur / new_price_eur * 100:.0f}% of new price")
    c2.caption(f"Model output on the dataset's log scale: {prediction:.2f} = ln(€{price_eur:,.0f})")

    for note in out_of_range(device, metadata):
        st.warning(f"Outside the training data – prediction is less reliable: {note}")

    with st.expander("About the model and its limitations"):
        st.markdown(
            f"""
- **Algorithm:** {metadata['algorithm']} (`{metadata['estimator_class']}`), selected as the lowest mean RMSE among 5
  algorithms in 10-fold cross validation; trained on all {metadata['trained_on_rows']:,} devices.
- **Cross-validation:** RMSE {cv['mean_rmse']:.3f} ± {cv['std_rmse']:.3f}, MAE {cv['mean_mae']:.3f}, R² {cv['mean_r2']:.3f}.
- **Data:** [{metadata['dataset']['title']}]({metadata['dataset']['url']}) (Kaggle, {metadata['dataset']['license']}),
  devices released 2013–2020.
- **Inputs (8):** brand, new price of the same model, storage, screen size, rear & front camera, release year, days used.
  OS, 4G/5G, RAM, battery and weight were left out to keep the form short (only +1.6 % RMSE).
- **Not included in the data (so not used):** model name, physical condition, battery health, seller, region.
- **Currency:** the dataset stores prices as a "normalized" value, which is the natural log of the price in euros, so
  the app converts with € = e^value. The Kaggle page does not state this. It is inferred from the data: e.g. a 2018
  Apple 6.5″ 64 GB phone has e^value = €1,250, matching the iPhone XS Max's €1,249 EU launch price.
"""
        )


if __name__ == "__main__":
    main()
