import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
DATA_FILE = Path("airlines_flights_data.csv")
MODEL_FILE = Path("model.pkl")
METRICS_FILE = Path("model_metrics.json")
HISTORY_FILE = Path("prediction_history.json")
TARGET = "price"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df = df.sample(frac=0.075, random_state=42)
    df = df.drop(columns=["index", "flight"], errors="ignore")
    return df.dropna()


def build_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        [
            ("cat", encoder, cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )

    return Pipeline(
        [
            ("prep", preprocessor),
            ("model", model),
        ]
    )


def train_and_save_model() -> Pipeline:
    print("Loading dataset...")
    df = load_dataset()
    print("Using dataset size:", df.shape)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    pipeline = build_pipeline(cat_cols, num_cols)

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

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
    print(f"CV R2:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    fitted_model = pipeline.named_steps["model"]
    fitted_prep = pipeline.named_steps["prep"]

    cat_feature_names = fitted_prep.transformers_[0][1].get_feature_names_out(cat_cols).tolist()
    all_feature_names = cat_feature_names + num_cols

    feat_imp = sorted(
        zip(all_feature_names, fitted_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    top_features = {name: float(imp) for name, imp in feat_imp[:15]}

    metrics = {
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std": round(cv_scores.std(), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_importance": top_features,
    }

    with METRICS_FILE.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)

    joblib.dump(pipeline, MODEL_FILE)
    print("Model and metrics saved")
    return pipeline


def load_metrics() -> dict:
    if not METRICS_FILE.exists():
        return {}
    try:
        with METRICS_FILE.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(history_items: list[dict]) -> None:
    with HISTORY_FILE.open("w", encoding="utf-8") as file_obj:
        json.dump(history_items, file_obj, ensure_ascii=False, indent=2)


def log_prediction(payload: dict, predicted_price: float) -> None:
    history = load_history()
    history_item = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "airline": payload.get("airline"),
        "source_city": payload.get("source_city"),
        "destination_city": payload.get("destination_city"),
        "departure_time": payload.get("departure_time"),
        "arrival_time": payload.get("arrival_time"),
        "stops": payload.get("stops"),
        "class": payload.get("class"),
        "duration": payload.get("duration"),
        "days_left": payload.get("days_left"),
        "predicted_price": round(float(predicted_price), 2),
    }
    history.insert(0, history_item)
    save_history(history[:100])


def get_feature_columns_from_model(model: Pipeline, data: pd.DataFrame) -> list[str]:
    try:
        prep = model.named_steps["prep"]
        cat_cols = list(prep.transformers_[0][2])
        num_cols = list(prep.transformers_[1][2])
        return cat_cols + num_cols
    except Exception:
        return [col for col in data.columns if col not in {TARGET, "index", "flight"}]


def build_options(data: pd.DataFrame) -> dict:
    return {
        "airline": sorted(data["airline"].dropna().unique().tolist()),
        "source_city": sorted(data["source_city"].dropna().unique().tolist()),
        "destination_city": sorted(data["destination_city"].dropna().unique().tolist()),
        "departure_time": sorted(data["departure_time"].dropna().unique().tolist()),
        "arrival_time": sorted(data["arrival_time"].dropna().unique().tolist()),
        "stops": sorted(data["stops"].dropna().unique().tolist()),
        "class": sorted(data["class"].dropna().unique().tolist()),
        "duration": {
            "min": float(data["duration"].min()),
            "max": float(data["duration"].max()),
        },
        "days_left": {
            "min": int(data["days_left"].min()),
            "max": int(data["days_left"].max()),
        },
    }


def build_input_frame(payload: dict, feature_columns: list[str]) -> pd.DataFrame:
    missing = [field for field in feature_columns if field not in payload]
    if missing:
        raise KeyError(", ".join(missing))

    row = {field: payload[field] for field in feature_columns}
    row["duration"] = float(row["duration"])
    row["days_left"] = int(row["days_left"])

    if row["source_city"] == row["destination_city"]:
        raise ValueError("Source and destination city cannot be the same")

    frame = pd.DataFrame([row])
    return frame[feature_columns]


def serve_api() -> None:
    data = pd.read_csv(DATA_FILE)
    if MODEL_FILE.exists():
        model = joblib.load(MODEL_FILE)
    else:
        print("model.pkl not found, training a new model first...")
        model = train_and_save_model()

    feature_columns = get_feature_columns_from_model(model, data)
    options = build_options(data)

    class PredictHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()

        def do_GET(self):
            if self.path == "/":
                self._send_json(
                    200,
                    {
                        "service": "flight-ticket-price-prediction-api",
                        "status": "ok",
                        "endpoints": ["/health", "/metadata", "/metrics", "/history", "/predict"],
                    },
                )
                return

            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return

            if self.path == "/metadata":
                self._send_json(200, {"options": options})
                return

            if self.path == "/metrics":
                self._send_json(200, {"metrics": load_metrics()})
                return

            if self.path == "/history":
                self._send_json(200, {"history": load_history()})
                return

            self._send_json(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/predict":
                self._send_json(404, {"error": "Not found"})
                return

            content_len = int(self.headers.get("Content-Length", 0))
            if content_len <= 0:
                self._send_json(400, {"error": "Missing JSON body"})
                return

            try:
                raw = self.rfile.read(content_len)
                payload = json.loads(raw.decode("utf-8"))
                input_df = build_input_frame(payload, feature_columns)
                prediction = float(model.predict(input_df)[0])
                log_prediction(payload, prediction)
                self._send_json(200, {"predicted_price": round(prediction, 2)})
            except KeyError as key_error:
                self._send_json(400, {"error": f"Missing field: {key_error}"})
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})

    server = HTTPServer((HOST, PORT), PredictHandler)
    print(f"Prediction API running at http://{HOST}:{PORT} (from train_model.py)")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model or run prediction API server")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model and save model.pkl + model_metrics.json",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run backend prediction API server",
    )
    args = parser.parse_args()

    if args.serve and not args.train:
        serve_api()
    elif args.train and args.serve:
        train_and_save_model()
        serve_api()
    else:
        train_and_save_model()