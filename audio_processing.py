import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the audio files
input_audio_dir = "./audio"
# Directory to save the processed audio features
output_audio_dir = "./processed_audio"

# Ensure the output directory exists
os.makedirs(output_audio_dir, exist_ok=True)

def load_audio(file_path):
    # Load the audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate

def extract_features(audio_data, sample_rate):
    # Extract MFCC features from the audio data
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    return mfccs

def save_features(features, file_path):
    # Save the extracted features to a file
    np.save(file_path, features)

def process_audio(file_path, output_file_path):
    audio_data, sample_rate = load_audio(file_path)
    features = extract_features(audio_data, sample_rate)
    save_features(features, output_file_path)
    print(f"Processed and saved features for {file_path}")

def main():
    for audio_file in os.listdir(input_audio_dir):
        if audio_file.endswith(".wav"):
            input_file_path = os.path.join(input_audio_dir, audio_file)
            output_file_path = os.path.join(output_audio_dir, f"{os.path.splitext(audio_file)[0]}.npy")
            process_audio(input_file_path, output_file_path)

if __name__ == "__main__":
    main()
