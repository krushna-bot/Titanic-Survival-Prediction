import numpy as np
import joblib  # or use pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
model = joblib.load("titanic_model.pkl")  # Ensure this file exists

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input from the form
        age = float(request.form["age"])
        gender = int(request.form["gender"])
        pclass = int(request.form["pclass"])
        fare = float(request.form["fare"])
        cabin = request.form["cabin"]

        # Convert cabin info into a meaningful feature
        cabin_feature = 1 if cabin.strip() else 0  # If cabin info is given, use 1 else 0

        # 🚨 Add missing features! Check your training data 🚨
        # Example: If your model was trained with these additional features
        sibsp = 0  # Number of siblings/spouses aboard (default if not provided)
        parch = 0  # Number of parents/children aboard (default if not provided)
        embarked_C = 0  # Embarked from C (Cherbourg)
        embarked_Q = 0  # Embarked from Q (Queenstown)
        embarked_S = 1  # Embarked from S (Southampton) - default if unknown

        # Create input array matching the training feature set
        input_features = np.array([[age, gender, pclass, fare, cabin_feature, sibsp, parch, embarked_C, embarked_Q, embarked_S]])

        # Ensure the model is an actual model (not a NumPy array)
        if hasattr(model, "predict"):
            prediction = model.predict(input_features)[0]  # Get the first prediction
            result_text = "Survived 🟢" if prediction == 1 else "Not Survived 🔴"
        else:
            return jsonify({"error": "Loaded object is not a valid model"})

        return render_template("index.html", prediction=result_text)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
