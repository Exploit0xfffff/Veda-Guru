import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score

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
    # Extract short-time signal energy
    energy = np.array([sum(abs(audio_data[i:i+512]**2)) for i in range(0, len(audio_data), 512)])
    # Extract sub-band energies
    sub_band_energies = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
    # Extract spectral flux
    spectral_flux = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    # Combine all features into a single feature vector
    features = np.hstack((energy, sub_band_energies.mean(axis=1), spectral_flux, mfccs.mean(axis=1)))
    return features

def save_features(features, file_path):
    # Save the extracted features to a file
    np.save(file_path, features)

def process_audio(file_path, output_file_path):
    audio_data, sample_rate = load_audio(file_path)
    features = extract_features(audio_data, sample_rate)
    save_features(features, output_file_path)
    print(f"Processed and saved features for {file_path}")

def load_features_and_labels():
    features = []
    labels = []
    for file_name in os.listdir(output_audio_dir):
        if file_name.endswith(".npy"):
            file_path = os.path.join(output_audio_dir, file_name)
            feature = np.load(file_path)
            label = 1 if "yes" in file_name else 0  # Assuming file names contain "yes" or "no" for labels
            features.append(feature)
            labels.append(label)
    return np.array(features), np.array(labels)

def train_classification_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    classifiers = {
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "NN": MLPClassifier()
    }
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        print(f"{name} - Recall: {recall}, Precision: {precision}")

def main():
    for audio_file in os.listdir(input_audio_dir):
        if audio_file.endswith(".wav"):
            input_file_path = os.path.join(input_audio_dir, audio_file)
            output_file_path = os.path.join(output_audio_dir, f"{os.path.splitext(audio_file)[0]}.npy")
            process_audio(input_file_path, output_file_path)

    features, labels = load_features_and_labels()
    train_classification_model(features, labels)

if __name__ == "__main__":
    main()
