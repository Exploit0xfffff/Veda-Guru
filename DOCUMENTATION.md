# Veda-Guru Project Documentation

## Overview
The Veda-Guru project aims to develop a machine learning model capable of processing and understanding Vedic texts and chants. The project involves gathering texts from the Rig Veda, Sama Veda, Yajur Veda, and Atharva Veda, preprocessing these texts, and training a model to handle both text and audio data. The model is designed to recognize special symbols in Vedic sentences and ensure unbiased predictions.

## Dataset
The dataset consists of texts from the four Vedas, sourced from Sacred-texts.com. The texts are available in both English and Sanskrit, with each verse as a separate file in UTF-8 Unicode Devanagari and standard romanization. Additionally, audio files of Vedic chants have been collected from YouTube and processed for model training.

### Text Data
- **Source**: Sacred-texts.com
- **Languages**: English, Sanskrit
- **Format**: UTF-8 Unicode Devanagari, standard romanization

### Audio Data
- **Source**: YouTube
- **Format**: .wav
- **Processing**: Extracted features using `librosa` and saved in the `processed_audio` directory

## Model Architecture
The model is a Convolutional Neural Network (CNN) designed to handle both text and audio data. The architecture includes layers for feature extraction, convolution, and dense layers for classification.

### Key Functions
- `load_audio`: Loads audio files for processing
- `extract_features`: Extracts MFCC features from audio data
- `preprocess_labels`: Preprocesses labels for training
- `build_model`: Builds the CNN model
- `train_model`: Trains the model on the dataset
- `evaluate_model`: Evaluates the model's performance

## Training Process
The training process involves loading the dataset, preprocessing the data, and training the model using the CNN architecture. Special care has been taken to ensure the model is not biased and can handle special symbols in Vedic sentences.

### Steps Taken to Ensure Unbiased Model
- **Diverse Dataset**: Ensured the dataset is diverse and representative of different styles and schools of Vedic chanting.
- **Special Symbols Handling**: Verified that preprocessing steps do not remove or alter special symbols in Vedic sentences.
- **Evaluation**: Rigorously evaluated the model's performance on a diverse set of audio samples.

## API for Interacting with the Trained Model
An API has been developed to interact with the trained model. The API includes a `POST /predict` endpoint that allows users to submit text or audio data and receive predictions from the model.

### API Endpoint
- **Endpoint**: `POST /predict`
- **Input**: Text or audio data
- **Output**: Model predictions

## Usage Instructions
1. **Clone the Repository**: `git clone https://github.com/kasinadhsarma/Veda-Guru.git`
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Run Preprocessing Scripts**: Execute the preprocessing scripts to prepare the dataset.
4. **Train the Model**: Run the `audio_mode_training.py` script to train the model.
5. **Start the API**: Use the provided API script to start the server and interact with the trained model.

## Development Process
The development process involved setting up the environment, scripting for text and audio processing, addressing errors, optimizing scripts for memory allocation, and ensuring the model is unbiased and can handle special symbols. The process also included rigorous evaluation and fine-tuning of the model, as well as developing an API for interaction.

## Conclusion
The Veda-Guru project successfully developed a machine learning model capable of processing and understanding Vedic texts and chants. The model is designed to handle special symbols in Vedic sentences and ensure unbiased predictions. The provided API allows users to interact with the trained model, making it accessible for various applications.

For any questions or further assistance, please refer to the project's GitHub repository or contact the project maintainers.
