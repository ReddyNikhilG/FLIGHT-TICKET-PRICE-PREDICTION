# ✈️ Flight Ticket Price Prediction System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)

## 📌 Project Overview
This is a high-performance **Flight Ticket Price Prediction Web Application**. The system uses historical flight data to predict the estimated fare of a flight based on various user-defined parameters. 

By analyzing patterns in airline demand, travel routes, and booking classes, the model provides travelers with data-backed price estimations to help them make better travel decisions.

---

## 🚀 Features
* **Machine Learning Powered:** Uses a Random Forest Regressor for highly accurate predictions.
* **Interactive Dashboard:** Built with **Streamlit** for a seamless user experience.
* **Advanced UI Components:** Includes a custom **TypeScript/React** frontend logic (located in `/flypredict-ui`).
* **Real-time Inference:** Instant price calculation upon entering flight details.
* **Data Visualization:** Comprehensive insights into feature importance and price trends.

---

## 🛠️ Technology Stack

### **Backend & Machine Learning**
* **Language:** Python
* **ML Library:** Scikit-learn (Random Forest Regressor)
* **Data Handling:** Pandas, NumPy
* **Serialization:** Pickle (for `model.pkl`)

### **Frontend**
* **Framework:** Streamlit
* **Languages:** TypeScript, JavaScript (Custom UI components)
* **Design:** CSS3, HTML5
* **Charts:** Plotly / Matplotlib

### **Dev Tools**
* **Deployment:** Render / Streamlit Cloud
* **Environment:** Docker / DevContainers

---

## 📊 Model Performance
The model has been rigorously evaluated to ensure reliability:

| Metric | Accuracy / Value |
| :--- | :--- |
| **R² Score** | **0.98** (Explains 98% of price variance) |
| **Mean Absolute Error (MAE)** | **₹1,100** |
| **Root Mean Squared Error (RMSE)** | **1,800** |

---

## 📂 Project Structure
```bash
FLIGHT-TICKET-PRICE-PREDICTION
├── flypredict-ui/           # TypeScript UI assets
├── .devcontainer/           # Dev environment settings
├── airlines_flights_data.csv # Raw Dataset
├── train_model.py           # ML Training Script
├── app.py                   # Main Streamlit Application
├── model.pkl                # Trained Serialized Model
├── model_metrics.json       # Performance evaluation logs
└── requirements.txt         # Project Dependencies

# 👨‍💻 Author

**Gali Reddy Nikhil**

B.Tech Computer Science (AI & ML)

GitHub
https://github.com/ReddyNikhilG

---

# 📜 License

This project is created for **educational and learning purposes**.
