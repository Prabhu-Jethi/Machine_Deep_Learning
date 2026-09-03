import uuid
import logging
import pandas as pd
import chromadb

class PortfolioDB:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        # Use native chromadb PersistentClient instead of LangChain wrapper for simpler access
        self.client = chromadb.PersistentClient('vectorstore')
        self.collection = self.client.get_or_create_collection(name="portfolio")
        
    def load_portfolio(self):
        """Loads data from the CSV into ChromaDB if the collection is empty."""
        # Only load if the database is empty to avoid duplicates
        if self.collection.count() > 0:
            return
        try:
            df = self.data
            documents = df["Techstack"].tolist()
            metadatas = [{"links": link} for link in df["Links"].tolist()]
            ids = [str(uuid.uuid4()) for _ in range(len(df))]
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        except Exception as e:
            return logging.error(f"Failed to load portfolio database: {e}")
             

    def query_links(self, skills):
        """Query the database for relevant portfolio links based on skills."""
        return self.collection.query(
            query_texts=skills,
            n_results=10
        ).get('metadatas', [])



if __name__ == "__main__":
    print("Initializing Database...")
    db = PortfolioDB(file_path=r"D:\Python\MachineLearning\EVERYDAY_ML_DL\GenAI\Cold_Email_generator\data\my_portfolio.csv")
    print("Loading Portfolio data...")
    db.load_portfolio()
    
    links = db.query_links("Python")
    
    print("\n--- Results ---")
    print(links)