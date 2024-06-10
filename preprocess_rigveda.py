import os
import re
import unicodedata
from indicnlp.tokenize import indic_tokenize

# Directory containing the downloaded hymns
input_dir = "."
# Directory to save the preprocessed hymns
output_dir = "./preprocessed"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def save_text(text, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

def remove_metadata(text):
    # Remove metadata and navigation links
    text = re.sub(r'Sacred Texts.*?Next', '', text, flags=re.DOTALL)
    text = re.sub(r'Next: Hymn \d+', '', text)
    return text.strip()

def normalize_text(text):
    # Normalize Unicode characters
    return unicodedata.normalize("NFC", text)

def tokenize_text(text):
    # Tokenize the text using Indic NLP Library
    return " ".join(indic_tokenize.trivial_tokenize(text))

def preprocess_text(file_path):
    text = load_text(file_path)
    text = remove_metadata(text)
    text = normalize_text(text)
    text = tokenize_text(text)
    return text

def main():
    for i in range(1, 6):
        input_file = os.path.join(input_dir, f"rigveda_hymn_{i}.txt")
        output_file = os.path.join(output_dir, f"rigveda_hymn_{i}.txt")
        preprocessed_text = preprocess_text(input_file)
        save_text(preprocessed_text, output_file)
        print(f"Hymn {i} preprocessed and saved.")

if __name__ == "__main__":
    main()
