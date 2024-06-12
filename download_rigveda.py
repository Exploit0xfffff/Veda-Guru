import requests
from bs4 import BeautifulSoup

# URLs of the Rig Veda hymns in Sanskrit
hymn_urls = [
    "https://sacred-texts.com/hin/rvsan/rv01001.htm",
    "https://sacred-texts.com/hin/rvsan/rv01002.htm",
    "https://sacred-texts.com/hin/rvsan/rv01003.htm",
    "https://sacred-texts.com/hin/rvsan/rv01004.htm",
    "https://sacred-texts.com/hin/rvsan/rv01005.htm"
    # Add more URLs as needed
]

def download_hymn(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        hymn_text = soup.get_text()
        return hymn_text
    else:
        return None

def save_hymn(hymn_text, hymn_number):
    with open(f"rigveda_hymn_{hymn_number}.txt", "w", encoding="utf-8") as file:
        file.write(hymn_text)

def main():
    for i, url in enumerate(hymn_urls, start=1):
        hymn_text = download_hymn(url)
        if hymn_text:
            save_hymn(hymn_text, i)
            print(f"Hymn {i} downloaded and saved.")
        else:
            print(f"Failed to download Hymn {i}.")

if __name__ == "__main__":
    main()