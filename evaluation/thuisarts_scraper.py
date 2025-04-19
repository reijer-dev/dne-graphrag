import os
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

class ThuisartsScraper:

    def download_documents(self, directory):
        """Loads all URL's from the Thuisarts website and scrapes them."""
        resp = requests.get('https://www.thuisarts.nl/sitemap.xml?page=1')
        root = ET.fromstring(resp.content)
        urls = [url[0].text for url in root]

        for url in urls[1:400]:
            self.__scrape_page(url, directory)
    
    def __scrape_page(self, url, directory):
        print(f'Scraping {url}')
        page = requests.get(url).content
        soup = BeautifulSoup(page, 'html.parser')
        page_title = soup.find('h1', class_='page-title').get_text(strip=True)
        blocks = soup.find_all('div', class_='field--name-field-ref-text-block')
        i = 0
        for block in blocks:
            h2 = block.find('h2')
            if h2 is not None:
                title = block.find('h2').get_text(strip=True)
                if title == "Film" or title == "Over deze tekst":
                    continue
            else:
                continue
            text = block.find('article').find('div')
            if text is not None:
                text = text.get_text(strip=True)
            else:
                continue

            if len(text) < 100:
                continue
            
            # Fix the nested quotes issue by using double quotes for the f-string
            cleaned_url = url.replace("https://www.thuisarts.nl/", "").replace("/", "")
            filename = f"{directory}/{cleaned_url}-{i}.txt"
            file = open(filename, "w", encoding="utf-8")
        
            document = f"Document title: {page_title}\nURL: {url}\nParagraph title: {title}\nText: {text}\n\n"
            file.write(document)
            i += 1
            file.close()

directory = "./evaluation/documents"
try:
    os.mkdir(directory)
    ThuisartsScraper().download_documents(directory)
except FileExistsError:
    print(f"Skipped scraping, directory already exists, delete the directory if you want to reset")

