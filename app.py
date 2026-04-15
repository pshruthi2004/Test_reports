from flask import Flask, request, jsonify, render_template
import pickle
import os

# ✅ Initialize Flask
app = Flask(__name__)

# ✅ Check if model exists
if not os.path.exists("model.pkl"):
    raise Exception("Model not found. Please run train.py first.")

# ✅ Load trained model
with open("model.pkl", "rb") as f:
    model, le_hospital, le_test, le_unit = pickle.load(f)

# ✅ Home route (loads HTML + CSS automatically)
@app.route("/")
def home():
    return render_template("index.html")

# ✅ API route (for Postman)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        h = le_hospital.transform([data['hospital']])[0]
        t = le_test.transform([data['test_name']])[0]
        u = le_unit.transform([data['unit']])[0]

        prediction = model.predict([[h, t, u]])

        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ Form route (HTML form submission)
@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        hospital = request.form['hospital']
        test = request.form['test_name']
        unit = request.form['unit']

        h = le_hospital.transform([hospital])[0]
        t = le_test.transform([test])[0]
        u = le_unit.transform([unit])[0]

        prediction = model.predict([[h, t, u]])

        return render_template("index.html", prediction=round(prediction[0], 2))

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

# ✅ Run app
if __name__ == "__main__":
    app.run(debug=True)