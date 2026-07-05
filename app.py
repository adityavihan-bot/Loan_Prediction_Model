from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/decision_tree.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    income = float(request.form["income"])
    loan_amount = float(request.form["loan_amount"])
    credit_score = float(request.form["credit_score"])

    data = pd.DataFrame({
        "Age":[age],
        "Income":[income],
        "LoanAmount":[loan_amount],
        "CreditScore":[credit_score]
    })

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "✅ Loan Approved"
    else:
        result = "❌ Loan Rejected"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)