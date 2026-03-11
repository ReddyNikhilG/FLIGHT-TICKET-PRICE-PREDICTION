# ✈️ Flight Ticket Price Prediction Web Application

## 📌 Project Overview

This project is a **Flight Ticket Price Prediction Web Application** built using **Python, Machine Learning, and Streamlit**.

The system predicts the **estimated price of a flight ticket** based on user inputs such as airline, source city, destination city, number of stops, class, and departure time.

Flight ticket prices vary due to factors such as airline demand, travel routes, time of booking, and number of stops. By analyzing historical flight data, machine learning models can identify patterns and predict approximate ticket prices.

This application provides a **simple and interactive web interface** where users can enter flight details and instantly receive the **predicted ticket price**.

---

# 🚀 Features

* Flight ticket price prediction using Machine Learning
* Interactive **Streamlit web interface**
* Real-time price prediction based on user input
* Data visualization and insights
* Feature importance visualization
* Airline price heatmap
* Flight price trend analysis
* Fast and lightweight application

---

# 🛠️ Technologies Used

## Backend

* Python
* Scikit-learn
* Pandas
* NumPy

---

## Machine Learning

* Random Forest Regressor
* Regression Algorithms
* Feature Engineering
* Model Training & Evaluation

---

## Frontend

* Streamlit
* HTML
* CSS
* Plotly / Matplotlib

---

## Tools

* Git
* GitHub
* Google Colab / Jupyter Notebook
* Visual Studio Code

---

# 📂 Project Structure

```
FLIGHT-TICKET-PRICE-PREDICTION
│
├── airline_flight_data.csv
├── flight_price_prediction.ipynb
├── streamlit_app.py
├── model.pkl
│
├── images
│
├── README.md
│
└── requirements.txt
```

---

# ⚙️ Installation and Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/ReddyNikhilG/FLIGHT-TICKET-PRICE-PREDICTION.git
```

---

## 2️⃣ Navigate to Project Folder

```
cd FLIGHT-TICKET-PRICE-PREDICTION
```

---

## 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 4️⃣ Run the Streamlit Application

```
streamlit run streamlit_app.py
```

---

# 🌐 Access the Application

After running the application, open your browser and go to:

```
http://localhost:8501
```

You will see the **Flight Ticket Price Prediction Dashboard** where you can enter flight details and get predicted prices.

---

# 📊 Example

### Input

```
Airline: IndiGo
Source: Delhi
Destination: Mumbai
Stops: 1
Class: Economy
Departure Time: Morning
```

### Output

```
Predicted Flight Price: ₹5,200
```

---

# 📊 Dataset

The dataset contains **historical flight booking information** including:

* Airline
* Source City
* Destination City
* Departure Time
* Arrival Time
* Stops
* Class
* Duration
* Ticket Price (Target Variable)

These features are used to train the machine learning model for predicting flight ticket prices.

---

# 📈 Model Performance

The machine learning model was evaluated using standard regression evaluation metrics.

| Metric                             | Description                                                    |
| ---------------------------------- | -------------------------------------------------------------- |
| **R² Score**                       | Measures how well the model explains variance in ticket prices |
| **MAE (Mean Absolute Error)**      | Average absolute difference between predicted and actual price |
| **RMSE (Root Mean Squared Error)** | Measures prediction error with higher penalty for large errors |

### Example Model Results

```
R² Score: 0.98
MAE: 1100
RMSE: 1800
```

These results indicate that the model can **predict flight prices with high accuracy**.

---

# 📊 Visualizations Included

The application includes several analytical visualizations:

* Airline vs Ticket Price Heatmap
* Feature Importance Graph
* Flight Price Trend Analysis
* Price Distribution Graphs

These visualizations help users understand **how different factors affect ticket prices**.

---

# 🎯 Use Cases

Flight ticket price prediction systems can be used for:

* Travel planning assistance
* Airline ticket price comparison
* Travel website recommendation systems
* Budget trip planning
* Airline market trend analysis

Such systems help travelers **find cheaper flights and make better travel decisions**.

---

# 🔮 Future Improvements

Possible improvements for this project include:

* Deep learning models for better prediction accuracy
* Integration with real-time airline APIs
* Price prediction for future travel dates
* Deployment on cloud platforms (AWS / GCP / Azure)
* Mobile-friendly interface

---

# 👨‍💻 Author

**Reddy Nikhil**

B.Tech Computer Science (AI & ML)

GitHub
https://github.com/ReddyNikhilG

---

# 📜 License

This project is created for **educational and learning purposes**.
