import logging
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_community.document_loaders import WebBaseLoader

class Scrape():
    def scrape_page_content(self, url):
        try:
            loader = WebBaseLoader(url)
            page_data = loader.load()
            return page_data[0].page_content if page_data else None
        
        except Exception as e:
            logging.error(f"Failed to fetch {url}: {e}")
            return None
    

if __name__ == "__main__":
    page_url = "https://careers.nike.com/software-engineer-iii-itc/job/R-49969"
    page_loader = Scrape()
    page_data = page_loader.scrape_page_content()
    print(page_data)