import requests
from bs4 import BeautifulSoup

# URLs of the Rig Veda Book 6 hymns in Sanskrit
hymn_urls = [
    "https://sacred-texts.com/hin/rvsan/rv06001.htm",
    "https://sacred-texts.com/hin/rvsan/rv06002.htm",
    "https://sacred-texts.com/hin/rvsan/rv06003.htm",
    "https://sacred-texts.com/hin/rvsan/rv06004.htm",
    "https://sacred-texts.com/hin/rvsan/rv06005.htm",
    "https://sacred-texts.com/hin/rvsan/rv06006.htm",
    "https://sacred-texts.com/hin/rvsan/rv06007.htm",
    "https://sacred-texts.com/hin/rvsan/rv06008.htm",
    "https://sacred-texts.com/hin/rvsan/rv06009.htm",
    "https://sacred-texts.com/hin/rvsan/rv06010.htm",
    "https://sacred-texts.com/hin/rvsan/rv06011.htm",
    "https://sacred-texts.com/hin/rvsan/rv06012.htm",
    "https://sacred-texts.com/hin/rvsan/rv06013.htm",
    "https://sacred-texts.com/hin/rvsan/rv06014.htm",
    "https://sacred-texts.com/hin/rvsan/rv06015.htm",
    "https://sacred-texts.com/hin/rvsan/rv06016.htm",
    "https://sacred-texts.com/hin/rvsan/rv06017.htm",
    "https://sacred-texts.com/hin/rvsan/rv06018.htm",
    "https://sacred-texts.com/hin/rvsan/rv06019.htm",
    "https://sacred-texts.com/hin/rvsan/rv06020.htm",
    "https://sacred-texts.com/hin/rvsan/rv06021.htm",
    "https://sacred-texts.com/hin/rvsan/rv06022.htm",
    "https://sacred-texts.com/hin/rvsan/rv06023.htm",
    "https://sacred-texts.com/hin/rvsan/rv06024.htm",
    "https://sacred-texts.com/hin/rvsan/rv06025.htm",
    "https://sacred-texts.com/hin/rvsan/rv06026.htm",
    "https://sacred-texts.com/hin/rvsan/rv06027.htm",
    "https://sacred-texts.com/hin/rvsan/rv06028.htm",
    "https://sacred-texts.com/hin/rvsan/rv06029.htm",
    "https://sacred-texts.com/hin/rvsan/rv06030.htm",
    "https://sacred-texts.com/hin/rvsan/rv06031.htm",
    "https://sacred-texts.com/hin/rvsan/rv06032.htm",
    "https://sacred-texts.com/hin/rvsan/rv06033.htm",
    "https://sacred-texts.com/hin/rvsan/rv06034.htm",
    "https://sacred-texts.com/hin/rvsan/rv06035.htm",
    "https://sacred-texts.com/hin/rvsan/rv06036.htm",
    "https://sacred-texts.com/hin/rvsan/rv06037.htm",
    "https://sacred-texts.com/hin/rvsan/rv06038.htm",
    "https://sacred-texts.com/hin/rvsan/rv06039.htm",
    "https://sacred-texts.com/hin/rvsan/rv06040.htm",
    "https://sacred-texts.com/hin/rvsan/rv06041.htm",
    "https://sacred-texts.com/hin/rvsan/rv06042.htm",
    "https://sacred-texts.com/hin/rvsan/rv06043.htm",
    "https://sacred-texts.com/hin/rvsan/rv06044.htm",
    "https://sacred-texts.com/hin/rvsan/rv06045.htm",
    "https://sacred-texts.com/hin/rvsan/rv06046.htm",
    "https://sacred-texts.com/hin/rvsan/rv06047.htm",
    "https://sacred-texts.com/hin/rvsan/rv06048.htm",
    "https://sacred-texts.com/hin/rvsan/rv06049.htm",
    "https://sacred-texts.com/hin/rvsan/rv06050.htm"
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
            save_hymn(hymn_text, f"book6_hymn_{i}")
            print(f"Hymn {i} downloaded and saved.")
        else:
            print(f"Failed to download Hymn {i}.")

if __name__ == "__main__":
    main()
