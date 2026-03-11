import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("Loading dataset...")

df = pd.read_csv("airlines_flights_data.csv")

# Take 5% random sample
df = df.sample(frac=0.05, random_state=42)

print("Using dataset size:", df.shape)

df = df.drop(columns=["index", "flight"], errors="ignore")

df = df.dropna()

TARGET = "price"

X = df.drop(columns=[TARGET])
y = df[TARGET]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    [
        ("cat", encoder, cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

pipeline = Pipeline(
    [
        ("prep", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print(f"Model R2:  {r2:.4f}")
print(f"MAE:       {mae:.2f}")
print(f"RMSE:      {rmse:.2f}")

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
print(f"CV R2:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Extract feature importance
fitted_model = pipeline.named_steps["model"]
fitted_prep = pipeline.named_steps["prep"]

cat_feature_names = fitted_prep.transformers_[0][1].get_feature_names_out(cat_cols).tolist()
all_feature_names = cat_feature_names + num_cols

importances = fitted_model.feature_importances_

feat_imp = sorted(
    zip(all_feature_names, importances),
    key=lambda x: x[1],
    reverse=True
)

# Save top 15 feature importances
top_features = {name: float(imp) for name, imp in feat_imp[:15]}

# Save metrics and feature importance
metrics = {
    "r2": round(r2, 4),
    "mae": round(mae, 2),
    "rmse": round(rmse, 2),
    "cv_r2_mean": round(cv_scores.mean(), 4),
    "cv_r2_std": round(cv_scores.std(), 4),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "feature_importance": top_features
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(pipeline, "model.pkl")

print("Model and metrics saved")