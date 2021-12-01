from PIL import Image
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from PredictionService import PredictionService

app = Flask(__name__)
load_dotenv()
app.prediction_service = PredictionService()


@app.route("/")
def hello():
    return "Backend up!"


@app.route("/predict", methods=['POST'])
def predict_image():
    try:
        image_file = request.files.get("image")
        image = Image.open(image_file)
        prediction = app.prediction_service.predict_image(image)

        return jsonify({"predicted_class": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
