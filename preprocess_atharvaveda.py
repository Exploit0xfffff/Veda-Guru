import os
import re
import unicodedata
from indicnlp.tokenize import indic_tokenize

# Directory containing the downloaded hymns
input_dir = "."
# Directory to save the preprocessed hymns
output_dir = "./preprocessed_atharvaveda"

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
    text = re.sub(r'Next: Book \d+', '', text)
    text = re.sub(r'Hymn \d+ : .*', '', text)  # Remove hymn titles
    text = re.sub(r'Previous.*?Next', '', text, flags=re.DOTALL)  # Remove additional navigation links
    text = re.sub(r'Page \d+', '', text)  # Remove page numbers
    text = re.sub(r'\[.*?\]', '', text)  # Remove text within square brackets
    text = re.sub(r'\bp\.a\b', '', text)  # Remove isolated page markers like "p.a"
    text = re.sub(r'\bp \. a\b', '', text)  # Remove additional page markers with different spacing
    text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)  # Remove line numbers
    text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r',? by Ralph T\. H\. Griffith( , ,)?( ,)? at sacred(-| )texts(\.| )com', '', text)  # Remove translator and website mentions
    text = re.sub(r',? by Ralph T\. H\. Griffith( , ,)?( ,)?', '', text)  # Remove additional translator mentions
    text = re.sub(r'General Navigation.*?Help ! Keep the Archive Alive !', '', text, flags=re.DOTALL)  # Remove general navigation and promotional content
    text = re.sub(r'Hold the world.*?palm of your hand', '', text, flags=re.DOTALL)  # Remove promotional content
    text = re.sub(r'Welcome to the largest.*?sacred-texts\.com', '', text, flags=re.DOTALL)  # Remove welcome message and promotional content
    text = re.sub(r'\bby Ralph T\. H\. Griffith\b', '', text)  # Remove additional translator mentions
    text = re.sub(r'\bat sacred-texts\.com\b', '', text)  # Remove additional website mentions
    text = re.sub(r'\bHelp ! Keep the Archive Alive !\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bGeneral Navigation\b', '', text)  # Remove additional navigation content
    text = re.sub(r'\bHymns of the Atharva Veda\b', '', text)  # Remove repeated book title
    text = re.sub(r'\bNext : Hymn \d+ : .*?\b', '', text)  # Remove next hymn navigation
    text = re.sub(r'\bNext : .*?\b', '', text)  # Remove next navigation
    text = re.sub(r'\bThe Internet Sacred Text Archive\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bUSB FLASH DRIVE\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bCD - ROM\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bQuran CD - ROM\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bISTA FLASH DRIVE\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bISTA\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bThe World Religions\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bMagick and Mystery\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bMyth and Folklore\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bOur Quran\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bThe Tarjuman al - Ashwaq\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bHome » Hinduism\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bVedas Upanishads Puranas Other Primary Texts Epics Mahabharata Ramayana Bhagavad Gita Vedanta Later texts Modern books\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bThere are four Vedas , the Rig Veda , Sama Veda , Yajur Veda and Atharva Veda\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bThe Vedas are the primary texts of Hinduism\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bThey also had a vast influence on Buddhism , Jainism , and Sikhism\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bTraditionally the text of the Vedas was coeval with the universe\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bScholars have determined that the Rig Veda , the oldest of the four Vedas , was composed about 1500 B . C\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bIt is unknown when it was finally committed to writing , but this probably was at some point after 300 B . C\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bThe Atharva Veda also contains material from the Rig Veda\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bbut of interest are the numerous incantations and metaphysical texts\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bwhich this anthology ( part of the Sacred Books of the East series ) collects and categorizes\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bThe Atharva Veda was written down much later than the rest of the Vedas\b', '', text)
    text = re.sub(r'about 200 B . C\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bit may have been composed about 1000 B . C\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bThe Hymns of the Atharvaveda translated by Ralph T . H . Griffith\b', '', text)
    text = re.sub(r'\bHelp! Keep the Archive Alive!\b', '', text)  # Remove additional promotional content
    text = re.sub(r'\bOmitted Hymn\b', '', text)  # Remove notes about omitted hymns
    text = re.sub(r'\bOmitted Verse\b', '', text)  # Remove notes about omitted verses
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
    for i in range(1, 21):
        input_file = os.path.join(input_dir, f"atharvaveda_book_{i}.txt")
        output_file = os.path.join(output_dir, f"atharvaveda_book_{i}.txt")
        preprocessed_text = preprocess_text(input_file)
        save_text(preprocessed_text, output_file)
        print(f"Book {i} preprocessed and saved.")

if __name__ == "__main__":
    main()
