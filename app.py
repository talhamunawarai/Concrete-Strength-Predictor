from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

with open("model.json", "r", encoding="utf-8") as f:
    model = json.load(f)

FEATURES = model["features"]
COEFFICIENTS = model["coefficients"]
INTERCEPT = model["intercept"]


def predict(values):
    return INTERCEPT + sum(c * x for c, x in zip(COEFFICIENTS, values))


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = [float(request.form[name]) for name in FEATURES]
            prediction = round(predict(values), 2)
        except (KeyError, ValueError):
            error = "Please enter a valid number in every field."

    return render_template(
        "index.html",
        features=FEATURES,
        prediction=prediction,
        error=error
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Send a JSON object with all model features."}), 400

    try:
        values = [float(data[name]) for name in FEATURES]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "All 8 features are required and must be numeric."}), 400

    prediction = predict(values)
    return jsonify({
        "prediction_mpa": round(prediction, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
