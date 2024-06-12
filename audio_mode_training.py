import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

def pad_features(features, max_length, num_mfcc):
    # Ensure the first dimension (number of MFCCs) is consistent
    if features.shape[0] != num_mfcc:
        raise ValueError(f"Inconsistent number of MFCCs: expected {num_mfcc}, got {features.shape[0]}")
    # Pad the features to ensure they all have the same length
    if features.shape[1] < max_length:
        pad_width = max_length - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    elif features.shape[1] > max_length:
        features = features[:, :max_length]
    return features

def load_features_and_labels():
    features = []
    labels = []
    lengths = []
    num_mfcc = 13  # Assuming 13 MFCCs

    # First pass: collect lengths of all feature arrays
    for file_name in os.listdir(output_audio_dir):
        if file_name.endswith(".npy") and file_name != "labels.npy":
            feature = np.load(os.path.join(output_audio_dir, file_name))
            lengths.append(feature.shape[1])

    # Calculate the 95th percentile length
    max_length = int(np.percentile(lengths, 95))
    print(f"Determined 95th percentile max_length: {max_length}")

    # Second pass: load features and labels, and pad features
    for file_name in os.listdir(output_audio_dir):
        if file_name.endswith(".npy") and file_name != "labels.npy":
            feature = np.load(os.path.join(output_audio_dir, file_name))
            print(f"Original feature shape for {file_name}: {feature.shape}")
            feature = pad_features(feature, max_length, num_mfcc)
            print(f"Padded feature shape for {file_name}: {feature.shape}")
            features.append(feature)
            # Assuming the label is part of the file name, e.g., "chanting_1.npy"
            label = file_name.split("_")[0]
            labels.append(label)

    # Save the max_length value to a file
    with open(os.path.join(output_audio_dir, "max_length.txt"), "w") as f:
        f.write(str(max_length))

    return np.array(features), np.array(labels)

def preprocess_labels(labels):
    if len(labels) == 0:
        raise ValueError("No labels found. Ensure that the output_audio_dir contains processed audio files with labels in their filenames.")
    # Convert labels to one-hot encoding
    label_set = sorted(set(labels))
    label_to_index = {label: index for index, label in enumerate(label_set)}
    one_hot_labels = np.array([label_to_index[label] for label in labels])
    return tf.keras.utils.to_categorical(one_hot_labels, num_classes=len(label_set))

def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def data_generator(features, labels, batch_size):
    num_samples = len(features)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_features = features[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]
            if len(batch_features) < batch_size:
                batch_features = np.pad(batch_features, ((0, batch_size - len(batch_features)), (0, 0), (0, 0), (0, 0)), mode='constant')
                batch_labels = np.pad(batch_labels, ((0, batch_size - len(batch_labels)), (0, 0)), mode='constant')
            yield batch_features, batch_labels

def main():
    # Process audio files
    for audio_file in os.listdir(input_audio_dir):
        if audio_file.endswith(".wav"):
            input_file_path = os.path.join(input_audio_dir, audio_file)
            output_file_path = os.path.join(output_audio_dir, f"{os.path.splitext(audio_file)[0]}.npy")
            process_audio(input_file_path, output_file_path)

    # Load features and labels
    features, labels = load_features_and_labels()
    features = np.expand_dims(features, axis=-1)  # Add channel dimension for Conv2D
    labels = preprocess_labels(labels)

    # Save the preprocessed labels to a file
    np.save(os.path.join(output_audio_dir, "labels.npy"), labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Build and train the model
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    model = build_model(input_shape, num_classes)
    train_generator = data_generator(X_train, y_train, batch_size=2)
    validation_generator = data_generator(X_test, y_test, batch_size=2)
    steps_per_epoch = len(X_train) // 2
    validation_steps = max(1, len(X_test) // 2)
    print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

    # Add callbacks for early stopping and model checkpoint
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath='./fine_tuned_model/model_checkpoint.keras', save_best_only=True)
    ]

    # Train the model with callbacks
    history = model.fit(
        train_generator,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1  # Add verbose output
    )

    # Save the training history
    np.save('./fine_tuned_model/training_history.npy', history.history)

    # Save the trained model
    model.save('./fine_tuned_model/model.h5', save_format='h5')
    print("Model saved successfully.")

def evaluate_model():
    # Load the pre-trained model
    model = tf.keras.models.load_model('./fine_tuned_model/model.h5')

    # Load the test set
    features, labels = load_features_and_labels()
    features = np.expand_dims(features, axis=-1)  # Add channel dimension for Conv2D
    labels = preprocess_labels(labels)
    _, X_test, _, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
    evaluate_model()
