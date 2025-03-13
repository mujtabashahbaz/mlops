from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

# Load Model and Tokenizer (Example: Hugging Face Model)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    # Tokenize and make prediction
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    return jsonify({"prediction": prediction})

@app.route("/")
def home():
    return "ML Model API is Running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
