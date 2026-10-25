# 📱 Smartphone Resale Price Predictor

COMSATS University Islamabad, Lahore Campus · Machine Learning Fundamentals (AIC354) · Assignment 3

A Streamlit web app that estimates the resale price of a **used or refurbished phone or tablet**. It uses a scikit-learn pipeline trained in `Used_Smartphone_Price_Prediction_Assignment_3.ipynb`.

**Live app:** _PENDING – add the Streamlit Community Cloud URL here after deployment._

## Dataset

| | |
|---|---|
| Title | Used Phones & Tablets Pricing Dataset |
| Source | https://www.kaggle.com/datasets/ahsan81/used-handheld-device-data (owner: Ahsan Raza, version 2, updated 2023-01-07) |
| Licence | CC0: Public Domain |
| File | `used_device_data.csv`: 278,314 bytes, SHA-256 `bf053b595d90db86770950630defa09fadfff09978ecd2e03e861d672fbad161` |
| Size | 3,454 devices × 15 columns (14 inputs + target). There are no duplicate rows, and no rows were removed during cleaning. |
| Target | `normalized_used_price` = ln(used price in €). The Kaggle page does not state the unit. We inferred it from the data: e^value for a 2018 Apple 6.5″ 64 GB phone is €1,250, matching the iPhone XS Max's €1,249 EU launch price. The app converts predictions to € with e^value. |

## Model

| | |
|---|---|
| Algorithms compared (10-fold CV, full data) | SVR, Extra Trees, AdaBoost, Histogram-based Gradient Boosting, MLP |
| Input features (8) | brand, new price of the same model (normalized), storage, screen size, rear camera, front camera, release year, days used |
| Selected (lowest mean RMSE) | **HistGradientBoostingRegressor**: RMSE 0.2298 ± 0.0113, MAE 0.1814, R² 0.846 |
| Saved artefact | `smartphone_price_model.joblib`: the complete pipeline (imputer + scaler + one-hot encoder + model) |
| Metadata | `model_metadata.json`: feature order, allowed categories, ranges, CV results, library versions |

The model uses 8 of the dataset's 14 attributes so the form stays short. With all 14 attributes the RMSE would be 0.2262, so dropping six costs only +1.6 %. The dataset has no model name, condition or battery-health columns, so the app does not ask for them.

## Files

```
app.py                         Streamlit application
smartphone_price_model.joblib  trained pipeline (scikit-learn 1.9.1)
model_metadata.json            schema + CV results used by the app
requirements.txt               pinned dependencies
.streamlit/config.toml         theme / server settings
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a **public GitHub repository** (for example `smartphone-resale-price-predictor`). Upload **all the files in this folder**, including the hidden `.streamlit` folder, to the repository root. You can use GitHub's "Add file → Upload files" page.
2. Go to https://share.streamlit.io and sign in with GitHub. Click **Create app → Deploy a public app from GitHub**.
3. Repository = your repository, Branch = `main`, Main file path = `app.py`.
4. Open **Advanced settings → Python version** and choose **3.12** or **3.13**. The pinned packages need Python ≥ 3.11.
5. Click **Deploy**. Wait for the build to finish, open the URL, and press **Predict resale price** to confirm a prediction appears.
6. Paste the URL at the top of the notebook (first cell) and at the top of this README.

> The model file was trained with scikit-learn **1.9.1**, and `requirements.txt` pins that version. If you retrain the model in Colab with a different scikit-learn version, either deploy the provided `smartphone_price_model.joblib` or change the scikit-learn pin to match Colab's version.

## Limitations

- Prices are 2021 euros, inferred from the data as described above. They are not PKR and not adjusted for inflation.
- The data covers devices released in 2013–2020 and was collected around 2021. Today's market may differ.
- Condition, battery health and model name are not available, so the model cannot account for them.
