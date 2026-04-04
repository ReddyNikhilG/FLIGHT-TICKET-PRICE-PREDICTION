# Flight Ticket Price Prediction 🚀

Predict flight ticket prices with machine learning! This project uses real-world airline data to build accurate regression models for dynamic pricing forecasts.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit-learn-1.2%2B-yellowgreen.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-orange.svg)](https://pandas.pydata.org/)

## 🎯 Problem Statement

Flight prices fluctuate based on airline, route, timing, stops, and seasonality. **Goal**: Predict prices with <10% MAPE.

## 📊 Dataset Overview

- **Source**: Kaggle (~10K records)
- **Key Features**: Airline, Source, Destination, Total_Stops, Date, Price
- **Target**: Price (₹1K-₹80K)

## 🛠 Complete Code Pipeline

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### 2. Data Loading & EDA
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/raw/flight_data.csv')
print(df.head())
print(df.info())
print(df.describe())

# Visualizations
plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
sns.boxplot(data=df, x='airline', y='price')
plt.xticks(rotation=45)
plt.title('Price by Airline')

plt.subplot(2,3,2)
sns.boxplot(data=df, x='total_stops', y='price')
plt.title('Price by Stops')

plt.subplot(2,3,3)
sns.scatterplot(data=df, x='days_left', y='price')
plt.title('Price vs Days Left')

plt.tight_layout()
plt.show()
```

### 3. Feature Engineering
```python
# Feature Engineering
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['days_left'] = (df['date'] - pd.Timestamp('2024-01-01')).dt.days

# Encode categorical variables
df['airline'] = df['airline'].astype('category').cat.codes
df['source_city'] = df['source_city'].astype('category').cat.codes
df['destination_city'] = df['destination_city'].astype('category').cat.codes
df['class'] = df['class'].map({'Economy': 0, 'Business': 1})

# Features and target
X = df.drop(['price', 'date', 'flight'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

### 4. Model Training & Evaluation
```python
# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"Random Forest - MAE: ₹{rf_mae:.0f}, R²: {rf_r2:.3f}")

# XGBoost (Best Model)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"XGBoost - MAE: ₹{xgb_mae:.0f}, R²: {xgb_r2:.3f}")

# Feature Importance
feat_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Features:")
print(feat_importance.head())
```

### 5. Model Results
