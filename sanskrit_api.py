from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_model/model.h5"
model = tf.keras.models.load_model(model_path)

# Function to load audio
def load_audio(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate

# Function to extract features
def extract_features(audio_data, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    return mfccs

# Function to pad features
def pad_features(features, max_length):
    if features.shape[1] < max_length:
        pad_width = max_length - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_length]
    return features

# Function to preprocess audio file
def preprocess_audio(file_path):
    audio_data, sample_rate = load_audio(file_path)
    features = extract_features(audio_data, sample_rate)

    # Load the max_length value from the saved file
    max_length_file_path = os.path.join('./processed_audio', 'max_length.txt')
    if os.path.exists(max_length_file_path):
        with open(max_length_file_path, 'r') as f:
            max_length = int(f.read().strip())
    else:
        raise FileNotFoundError(f"max_length.txt file not found at {max_length_file_path}")

    features = pad_features(features, max_length)
    features = np.expand_dims(features, axis=-1)  # Add channel dimension for Conv2D
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    return features

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio_file']
    if not audio_file.filename.endswith('.wav'):
        return jsonify({'error': 'Invalid file type. Only .wav files are supported.'}), 400

    try:
        audio_file_path = os.path.join('/tmp', audio_file.filename)
        audio_file.save(audio_file_path)

        # Preprocess the audio file
        features = preprocess_audio(audio_file_path)

        # Predict using the model
        predictions = model.predict(features)
        predicted_label = np.argmax(predictions, axis=-1)[0]

        return jsonify({'predicted_label': int(predicted_label)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
