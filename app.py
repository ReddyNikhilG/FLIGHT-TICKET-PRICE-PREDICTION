import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import joblib
import pandas as pd

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


def ensure_trained_model() -> None:
    if os.path.exists("model.pkl"):
        return
    subprocess.run([sys.executable, "train_model.py"], check=True)


def load_model_and_data():
    ensure_trained_model()
    model = joblib.load("model.pkl")
    data = pd.read_csv("airlines_flights_data.csv")
    return model, data


MODEL, DATA = load_model_and_data()


def infer_feature_columns() -> list[str]:
    # Prefer exact feature ordering from the trained pipeline.
    try:
        prep = MODEL.named_steps["prep"]
        cat_cols = list(prep.transformers_[0][2])
        num_cols = list(prep.transformers_[1][2])
        return cat_cols + num_cols
    except Exception:
        return [col for col in DATA.columns if col not in {"price", "index", "flight"}]


FEATURE_COLUMNS = infer_feature_columns()

OPTIONS = {
    "airline": sorted(DATA["airline"].dropna().unique().tolist()),
    "source_city": sorted(DATA["source_city"].dropna().unique().tolist()),
    "destination_city": sorted(DATA["destination_city"].dropna().unique().tolist()),
    "departure_time": sorted(DATA["departure_time"].dropna().unique().tolist()),
    "arrival_time": sorted(DATA["arrival_time"].dropna().unique().tolist()),
    "stops": sorted(DATA["stops"].dropna().unique().tolist()),
    "class": sorted(DATA["class"].dropna().unique().tolist()),
    "duration": {
        "min": float(DATA["duration"].min()),
        "max": float(DATA["duration"].max()),
    },
    "days_left": {
        "min": int(DATA["days_left"].min()),
        "max": int(DATA["days_left"].max()),
    },
}


def build_input_frame(payload: dict) -> pd.DataFrame:
    missing = [field for field in FEATURE_COLUMNS if field not in payload]
    if missing:
        raise KeyError(", ".join(missing))

    row = {field: payload[field] for field in FEATURE_COLUMNS}
    row["duration"] = float(row["duration"])
    row["days_left"] = int(row["days_left"])

    if "source_city" in row and "destination_city" in row and row["source_city"] == row["destination_city"]:
        raise ValueError("Source and destination city cannot be the same")

    frame = pd.DataFrame([row])
    return frame[FEATURE_COLUMNS]


class Handler(BaseHTTPRequestHandler):
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
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/metadata":
            self._send_json(200, {"options": OPTIONS})
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
            input_df = build_input_frame(payload)
            prediction = float(MODEL.predict(input_df)[0])
            self._send_json(200, {"predicted_price": round(prediction, 2)})
        except KeyError as key_error:
            self._send_json(400, {"error": f"Missing field: {key_error}"})
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Backend running at http://{HOST}:{PORT}")
    server.serve_forever()
