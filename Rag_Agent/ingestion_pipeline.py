import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

docs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

def load_document(docs_path=docs):
    ## Load all document's text files from docs directory
    print(f"Loading documents from {docs_path}..")

    ## check if directory is availabale or not
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} is not found or doesn't exist")
            
    ## Load all the docs_text_files 
    loader = DirectoryLoader(
        path=docs_path,
        loader_cls=TextLoader,
        glob="*.txt",
        loader_kwargs={'encoding': 'utf-8'}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No such `.txt` files found in {docs_path} directory.")
        
    ## show 1st 2 documents
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")

    return documents
    

def split_documents(documents):
    ## Splitting documents into smaller chunks having no overlaps
    text_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=0
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"Chunk {i+1}")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks


def vector_embeddings(chunks, persist_directory="db/chroma_db"):
    ## Convert these chunks into vector representation aka. vector embeddings using `Embedding models`
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    ## creating vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore


def main():
    '''Document Loader'''
    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    if os.path.exists(persistent_directory):
        print(f"Vector store already exists. No need to re-process documents")

        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vectorstore = Chroma(
            embedding_function=embedding_model,
            persist_directory=persistent_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore
    
    print("Persistent directory does not exist. Initializing vector store...\n")


    ## load documents
    documents = load_document(docs_path)
    
    ## chunks of documents using text splitter
    chunks = split_documents(documents)
    
    ## create vector store
    vectorstore = vector_embeddings(chunks, persistent_directory)

    return vectorstore


if __name__ == "__main__":
    main()