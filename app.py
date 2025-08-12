from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("bike_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            season = int(request.form["season"])
            holiday = int(request.form["holiday"])
            workingday = int(request.form["workingday"])
            weather = int(request.form["weather"])
            temp = float(request.form["temp"])
            humidity = float(request.form["humidity"])
            windspeed = float(request.form["windspeed"])

            # Prepare input for prediction
            input_data = np.array([[season, holiday, workingday, weather, temp, humidity, windspeed]])
            prediction = model.predict(input_data)[0]
            prediction = round(prediction, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
