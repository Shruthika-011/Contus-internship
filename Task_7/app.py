from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

def encode_input(data):
    values = []

    for key in ["gender","SeniorCitizen","Partner","Dependents",
                "tenure","PhoneService","InternetService",
                "MonthlyCharges","TotalCharges"]:
        
        if key in encoders:
            le = encoders[key]
            values.append(le.transform([data[key]])[0])
        else:   
            values.append(float(data[key]))

    return values

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()

        values = encode_input(form_data)

        data = np.array(values).reshape(1, -1)
        data = scaler.transform(data)

        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1] * 100

        if pred == 1:
            result = f"⚠️ High Risk ({prob:.2f}% chance of churn)"
        else:
            result = f"✅ Low Risk ({100-prob:.2f}% chance of staying)"

        return render_template("index.html", prediction=result, prob=prob)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)