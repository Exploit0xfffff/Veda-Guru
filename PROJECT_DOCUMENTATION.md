# Veda-Guru Project Documentation

## Project Overview

Veda-Guru is a Sanskrit language model specifically designed for the Vedas, including the Rigveda, Samaveda, Yajurveda, and Atharvaveda. The project aims to provide a comprehensive tool for understanding and analyzing Vedic texts, with a focus on audio mode training and handling special symbols in Vedic sentences.

## Key Steps

1. **Text Collection and Preprocessing**: Downloading and preprocessing texts from the Vedas.
2. **Model Training**: Fine-tuning a pre-trained BERT model on the Vedic texts.
3. **Evaluation and Fine-tuning**: Assessing the model's performance and making necessary adjustments.
4. **API Development**: Creating an API for interacting with the trained model.
5. **Audio Mode Training**: Incorporating techniques for decoding audio from different parts of the throat and handling special symbols in Vedic sentences.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/kasinadhsarma/Veda-Guru.git
    cd Veda-Guru
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install additional dependencies for audio and video processing:
    ```bash
    sudo apt-get install ffmpeg
    pip install librosa pydub wave pyaudio opencv-python moviepy
    ```

4. Set up Git LFS for handling large audio files:
    ```bash
    git lfs install
    git lfs track "*.wav"
    git add .gitattributes
    git commit -m "Track .wav files with Git LFS"
    ```

## Usage

### API Endpoints

The Veda-Guru API provides the following endpoints:

- `POST /predict`: Predicts the next tokens in a given input text.

### Training the Model

To train the model, run the following script:
```bash
python train_sanskrit_model.py
```

### Preprocessing

To preprocess the Vedic texts, run the following script:
```bash
python preprocess_rigveda.py
```

### Downloading Data

To download the Rigveda hymns, run the following script:
```bash
python download_rigveda.py
```

## Audio Mode Training

To train the audio model, run the following script:
```bash
python audio_mode_training.py
```

## API for Audio Predictions

The Veda-Guru API also provides an endpoint for audio predictions:

- `POST /predict`: Receives an audio file, processes it, and returns a prediction.

### Example Usage

To use the API for audio predictions, send a POST request with a `.wav` audio file to the `/predict` endpoint. The API will return the predicted label.

Example:
```bash
curl -X POST -F "audio_file=@./audio/vedic_chanting_Vedic Chanting ｜ Rudri Path by 21 Brahmins.wav" http://localhost:5000/predict
```

### Error Handling

The API includes error handling for common issues that users might encounter. Here are some examples:

- **Invalid File Format**: If the uploaded file is not a `.wav` file, the API will return a `400 Bad Request` error with a message indicating the invalid file format.
- **Missing File**: If no file is uploaded, the API will return a `400 Bad Request` error with a message indicating that the file is missing.
- **Internal Server Error**: If there is an issue processing the file or making a prediction, the API will return a `500 Internal Server Error` with a message indicating the problem.

### Model Capabilities and Limitations

The Veda-Guru model is designed to handle audio files of Vedic chanting and make predictions based on the trained data. However, there are some limitations to be aware of:

- **Special Symbols**: The model is trained to handle special symbols in Vedic sentences, but its performance may vary depending on the quality and clarity of the audio.
- **Bias**: Efforts have been made to ensure the model is not biased, but users should be aware that the training data's diversity can impact the model's predictions.
- **Audio Quality**: The model performs best with high-quality audio recordings. Poor audio quality may affect the accuracy of the predictions.

## Handling Special Symbols

The Veda-Guru model is designed to handle special symbols in Vedic sentences. During preprocessing, special symbols are retained and appropriately tokenized to ensure the model can accurately interpret and process them.

## Ensuring Model Unbias

Efforts have been made to ensure the Veda-Guru model is not biased. The training data includes a diverse range of examples to minimize bias. Additionally, the model's predictions are regularly evaluated to identify and address any potential biases.

## Interacting with the Trained Model

The trained model can be interacted with through the provided API. The `POST /predict` endpoint allows users to send audio files for prediction. The model is saved in the Keras format (`.keras` extension) and can be loaded using the following code snippet:
```python
from tensorflow.keras.models import load_model

model = load_model('fine_tuned_model/model.keras')
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
