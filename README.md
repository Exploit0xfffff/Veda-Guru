# Veda-Guru

Veda-Guru is a Sanskrit language model specifically designed for the Vedas, including the Rigveda, Samaveda, Yajurveda, and Atharvaveda. The project aims to provide a comprehensive tool for understanding and analyzing Vedic texts, with a focus on audio mode training and handling special symbols in Vedic sentences.

## Project Overview

The Veda-Guru project involves the following key steps:
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

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
