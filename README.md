# 🏠 Indian House Price Prediction

A machine learning project that predicts residential property prices in India using a **Random Forest Regressor**.

The project covers data cleaning, exploratory data analysis, categorical encoding, train/test splitting, regression modeling, evaluation, feature-importance analysis, and model serialization for deployment.

## 📌 Project Overview

The goal is to estimate a property's price in **lakhs (₹ Lakhs)** based on features such as:

- Property area
- Number of BHKs
- Property location
- RERA approval
- Construction status
- Ready-to-move status
- Resale status
- Latitude and longitude
- Property layout/type

## 📊 Dataset

The notebook uses the **House Price Prediction Challenge** dataset from Kaggle.

Dataset identifier used in the notebook:

`anmolkumar/house-price-prediction-challenge`

The target variable is:

`TARGET(PRICE_IN_LACS)`

## 🔧 Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle
- KaggleHub
- Jupyter Notebook

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset.
2. Checked dataset shape and information.
3. Checked for missing values.
4. Checked and removed duplicate records.
5. Examined categorical distributions.
6. Performed descriptive statistical analysis.
7. Visualized the data using box plots.
8. Applied one-hot encoding to:
   - `BHK_OR_RK`
   - `ADDRESS`
9. Removed:
   - `POSTED_BY`
   - `TARGET(PRICE_IN_LACS)` from the input features.

## 🤖 Machine Learning Model

### Random Forest Regressor

The final model uses:

```python
RandomForestRegressor(
    n_estimators=40,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

The model was trained using an **80/20 train-test split** with `random_state=43`.

## 📈 Model Performance

Evaluation on the test set:

| Metric | Score |
|---|---:|
| **R² Score** | **0.9491 (94.91%)** |
| **MAE** | **27.79 Lakhs** |
| **RMSE** | **123.36 Lakhs** |

### Train vs Test R²

| Dataset | R² |
|---|---:|
| Training | **0.9850 (98.50%)** |
| Testing | **0.9491 (94.91%)** |

The relatively small train-test R² gap indicates that the model does not show severe overfitting on the current test split.

> **Note:** RMSE is substantially higher than MAE because RMSE penalizes large prediction errors more heavily. This should be considered when interpreting the model's performance.

## 🔍 Feature Importance

Random Forest feature importance was analyzed after training.

Because `ADDRESS` was one-hot encoded, it produces multiple features such as:

```text
ADDRESS_Bangalore
ADDRESS_Chennai
ADDRESS_Pune
...
```

For original-column analysis, the importance values of all `ADDRESS_*` features were aggregated back into a single `ADDRESS` feature.

This provides a cleaner view of the importance of the original dataset columns.

## 💾 Model Export

The trained model is serialized using Pickle:

```python
pickle.dump(model, open("House-Price-Prediction-RFR.pkl", "wb"))
```

The saved model can be loaded later for prediction:

```python
import pickle

with open("House-Price-Prediction-RFR.pkl", "rb") as file:
    model = pickle.load(file)
```

## 🧪 Prediction Workflow

The overall workflow is:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Duplicate Removal
     ↓
EDA & Statistical Analysis
     ↓
Categorical Encoding
     ↓
Feature / Target Separation
     ↓
Train-Test Split
     ↓
Random Forest Regressor
     ↓
Prediction
     ↓
MAE / RMSE / R² Evaluation
     ↓
Feature Importance
     ↓
Model Serialization
     ↓
Deployment
```

## 🖥️ Deployment

The trained Random Forest model can be integrated into a web application where users enter property details and receive an estimated property price.

The prediction should be treated as a **machine-learning estimate**, not as a verified market valuation.

## ⚠️ Limitations

- Predictions depend heavily on the quality and distribution of the training data.
- Location is represented through one-hot encoded addresses, which can create a large number of features.
- RMSE is considerably higher than MAE, indicating the presence of larger prediction errors/outliers.
- A high overall R² does not guarantee that every individual property prediction is accurate.
- The model should not be treated as a substitute for professional property valuation.

## 🚀 Future Improvements

- Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
- Compare Random Forest with Gradient Boosting and XGBoost Regressor
- Perform cross-validation
- Investigate large-error/outlier properties
- Improve location feature engineering
- Extract city/locality information from addresses
- Use permutation importance for more robust feature analysis
- Optimize the serialized model for deployment
- Add prediction intervals or uncertainty estimates
- Deploy the model with Flask or Streamlit

## 📁 Project Structure

```text
Indian-House-Price-Prediction/
│
├── Indian_House_Prediction.ipynb
├── House-Price-Prediction-RFR.pkl
├── README.md
└── app.py                  # Optional deployment application
```

## 👨‍💻 Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Categorical Encoding
- Regression
- Ensemble Learning
- Random Forest
- Model Evaluation
- Feature Importance Analysis
- Model Serialization
- Machine Learning Deployment

## ⭐ Results

The final Random Forest model achieved a **94.91% R² score on the test set**, demonstrating strong predictive performance on the current dataset and test split.

If you found this project useful, consider giving the repository a ⭐.
