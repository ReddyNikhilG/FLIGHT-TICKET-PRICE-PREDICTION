# ✈️ FlyPredict — Flight Ticket Price Prediction System

<p align="center">
  <b>Predict flight ticket prices using Machine Learning with high accuracy 🚀</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/React-18.0+-61dafb.svg" alt="React Version">
  <img src="https://img.shields.io/badge/Vite-4.0+-646CFF.svg" alt="Vite Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---

## 📌 Project Overview  

**FlyPredict** is an advanced **Machine Learning-based web application** that predicts flight ticket prices based on multiple factors such as airline, route, duration, class, and booking time.

Built with a complete ML pipeline and deployed using a modern web stack, this system helps users make **smarter travel decisions** by forecasting ticket prices with high accuracy.

---

## 🌟 Key Highlights  

- 🎯 **96.8% R² Score** (High Accuracy Model)  
- 📊 Trained on **22,000+ flight records**  
- ⚡ Real-time price prediction  
- 📈 Interactive analytics dashboard  
- 🧠 End-to-end ML pipeline  
- 🌐 Full-stack web application (Python Backend + React UI)  

---

## 🚀 Features  

- 🔍 **Predict flight prices instantly** based on user input
- 📊 **Visual dashboard** showcasing model performance
- 🧠 **Smart feature engineering** for optimal predictions
- 📉 **Feature importance analysis** to understand key pricing factors
- 📜 **Prediction history tracking** to compare past searches
- ⚡ **Fast & responsive UI** built with React, Vite, and Tailwind CSS

---

## 🧠 Machine Learning Pipeline  

1. **Data Collection** – 22,000+ flight records sourced from Kaggle  
2. **EDA & Cleaning** – Handling missing values, outliers, and formatting  
3. **Feature Engineering** – Categorical encoding, scaling, and feature selection  
4. **Model Training** – Training Gradient Boosting Regressor to find the best fit  
5. **Evaluation** – Cross-validation and performance metrics assessment  
6. **Deployment** – Exposing the model via custom Python HTTP server API with a React frontend  

---

## 📈 Model Performance  

| Metric | Value |
|--------|-------|
| **R² Score** | **0.9678** |
| **MAE** | ₹2,313 |
| **RMSE** | ₹4,116 |
| **Cross-validation R²** | 0.9685 |
| **Std Deviation** | ±0.0013 |

> ✅ **Gradient Boosting Regressor** performed the best among all tested models.

---

## 🛠️ Tech Stack  

### 💻 Backend  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### 🤖 Machine Learning  
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

### 🌐 Frontend  
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Recharts](https://img.shields.io/badge/Recharts-%2322B5BF.svg?style=for-the-badge)

---

## 📂 Project Structure  
```text
FLIGHT-TICKET-PRICE-PREDICTION/
│
├── flypredict-ui/               # React UI (Frontend built with Vite)
├── airlines_flights_data.csv    # Dataset
├── app.py                       # Main application entry (Backend API)
├── train_model.py               # Script to train ML model & serve API
├── model.pkl                    # Trained Machine Learning model
├── model_metrics.json           # Metrics & feature importances from training
├── prediction_history.json      # Logs of past price predictions
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ▶️ How to Run Locally  

### 1️⃣ Clone Repository  
```bash
git clone https://github.com/ReddyNikhilG/FLIGHT-TICKET-PRICE-PREDICTION.git
cd FLIGHT-TICKET-PRICE-PREDICTION
```

### 2️⃣ Setup Backend
Install the dependencies and start the backend API:
```bash
pip install -r requirements.txt
python app.py
```
*(If the model is not trained yet, it will automatically train a new model when starting).*

### 3️⃣ Setup Frontend
Open a new terminal window and run:
```bash
cd flypredict-ui
npm install
npm run dev
```

---

## 🎯 Use Cases  

- ✈️ **Travelers** planning the cheapest booking time  
- 🏢 **Airlines** optimizing pricing strategies based on trends  
- 📊 **Students & Researchers** learning ML pipelines  

---

## 🔮 Future Enhancements  

- [ ] 🌍 Real-time flight API integration  
- [ ] 📱 Mobile responsive improvements  
- [ ] ☁️ Cloud deployment (AWS / Azure / Vercel / Render)  
- [ ] 🤖 Advanced models (XGBoost, Deep Learning)  

---

## 🤝 Contributing  

Contributions are welcome!  
Feel free to **fork** this repository, create a **feature branch**, and submit a **pull request**.

---

## 📜 License  

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author  

**Reddy Nikhil G**  
🔗 GitHub: [@ReddyNikhilG](https://github.com/ReddyNikhilG)  
