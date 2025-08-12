# Bike Sharing Demand Predictor

A Flask-based machine learning web app that forecasts bike rental demand using environmental, temporal, and historical data.

---

## Project Overview

This project predicts hourly bike-sharing demand using features like:
- Weather data (temperature, humidity, etc.)
- Time of day, day of the week, and seasonal factors
- Historical usage patterns

Built with Python, Flask, and scikit-learn (or your ML library of choice), the app helps bike operators balance availability more effectively.

---

## Project Structure

```
Bike_Sharing_Demand_Predictor/
├── app.py
├── model.pkl
├── requirements.txt
├── templates/
│   └── index.html
└── README.md
```

---

## Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/AbikshaaDevi/Bike_Sharing_Demand_Predictor.git
   cd Bike_Sharing_Demand_Predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000/` in your browser.

4. **Get a prediction**
   Enter relevant inputs (e.g., weather, day, time) in the form to get real-time demand predictions.

---

## Technologies Used

- **Flask** — Web server
- **scikit-learn** (or similar) — ML model
- **HTML templates** — UI layer
- **Python** — Backend logic

---

## Deployment Options

You can deploy it on platforms like:
- Heroku
- Render
- Any Python-compatible hosting service

---

## Expand and Improve

- **Add more data sources** (e.g., real-time weather APIs)
- **Compare different ML models** (Random Forest, XGBoost, etc.)
- **Add CSS or UI frameworks** (Bootstrap, Tailwind, etc.)
- **Improve performance metrics** (MAE, RMSE, R²)

---

## Author

**Abikshaa Devi** — Passionate about bridging machine learning with practical web applications to solve urban mobility challenges.

---

## License

Open-source and free to use!
