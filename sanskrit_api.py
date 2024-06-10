from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForMaskedLM
import torch

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_dir = "./fine_tuned_model"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForMaskedLM.from_pretrained(model_dir)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('input_text', '')

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted tokens
    predictions = torch.argmax(outputs.logits, dim=-1)
    predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)

    return jsonify({'predicted_text': predicted_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
