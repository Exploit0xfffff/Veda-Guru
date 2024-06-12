import requests
from bs4 import BeautifulSoup

# URLs of the Atharva Veda books in Sanskrit
book_urls = [
    "https://sacred-texts.com/hin/av/avbook01.htm",
    "https://sacred-texts.com/hin/av/avbook02.htm",
    "https://sacred-texts.com/hin/av/avbook03.htm",
    "https://sacred-texts.com/hin/av/avbook04.htm",
    "https://sacred-texts.com/hin/av/avbook05.htm",
    "https://sacred-texts.com/hin/av/avbook06.htm",
    "https://sacred-texts.com/hin/av/avbook07.htm",
    "https://sacred-texts.com/hin/av/avbook08.htm",
    "https://sacred-texts.com/hin/av/avbook09.htm",
    "https://sacred-texts.com/hin/av/avbook10.htm",
    "https://sacred-texts.com/hin/av/avbook11.htm",
    "https://sacred-texts.com/hin/av/avbook12.htm",
    "https://sacred-texts.com/hin/av/avbook13.htm",
    "https://sacred-texts.com/hin/av/avbook14.htm",
    "https://sacred-texts.com/hin/av/avbook15.htm",
    "https://sacred-texts.com/hin/av/avbook16.htm",
    "https://sacred-texts.com/hin/av/avbook17.htm",
    "https://sacred-texts.com/hin/av/avbook18.htm",
    "https://sacred-texts.com/hin/av/avbook19.htm",
    "https://sacred-texts.com/hin/av/avbook20.htm"
]

def download_hymn(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        hymn_content = soup.find_all("p")
        if hymn_content:
            hymn_text = "\n".join([p.get_text(separator="\n") for p in hymn_content])
            return hymn_text
        else:
            return None
    else:
        return None

def download_book(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        hymn_links = soup.find_all("a", href=True)
        book_text = ""
        for link in hymn_links:
            hymn_url = "https://sacred-texts.com/hin/av/" + link['href']
            hymn_text = download_hymn(hymn_url)
            if hymn_text:
                book_text += hymn_text + "\n\n"
        return book_text
    else:
        return None

def save_book(book_text, book_number):
    with open(f"atharvaveda_book_{book_number}.txt", "w", encoding="utf-8") as file:
        file.write(book_text)

def main():
    for i, url in enumerate(book_urls, start=1):
        book_text = download_book(url)
        if book_text:
            save_book(book_text, i)
            print(f"Book {i} downloaded and saved.")
        else:
            print(f"Failed to download Book {i}.")

if __name__ == "__main__":
    main()
