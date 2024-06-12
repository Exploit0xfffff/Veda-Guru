import requests
from bs4 import BeautifulSoup

# URLs of the Yajur Veda Kandas in Sanskrit
kanda_urls = [
    "https://sacred-texts.com/hin/yv/yv01.htm",
    "https://sacred-texts.com/hin/yv/yv02.htm",
    "https://sacred-texts.com/hin/yv/yv03.htm",
    "https://sacred-texts.com/hin/yv/yv04.htm",
    "https://sacred-texts.com/hin/yv/yv05.htm",
    "https://sacred-texts.com/hin/yv/yv06.htm",
    "https://sacred-texts.com/hin/yv/yv07.htm"
]

def download_kanda(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        kanda_text = soup.get_text()
        return kanda_text
    else:
        return None

def save_kanda(kanda_text, kanda_number):
    with open(f"yajurveda_kanda_{kanda_number}.txt", "w", encoding="utf-8") as file:
        file.write(kanda_text)

def main():
    for i, url in enumerate(kanda_urls, start=1):
        kanda_text = download_kanda(url)
        if kanda_text:
            save_kanda(kanda_text, i)
            print(f"Kanda {i} downloaded and saved.")
        else:
            print(f"Failed to download Kanda {i}.")

if __name__ == "__main__":
    main()
