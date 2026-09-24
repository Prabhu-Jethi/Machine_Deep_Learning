import os
from langchain_chroma import Chroma
from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from Ingestion import partition_document, create_chunks_by_title, summarized_chunks, vector_store, export_chunks_to_json
from dotenv import load_dotenv

load_dotenv()

file_path = "../docs/attention-is-all-you-need.pdf"

def complete_ingestion_pipeline(file_path: str) -> str:
    ## Run the complete RAG Ingestion pipeline
    print(f"Starting Rag ingestion pipeline")

    '''1. Partition document elements'''
    elements = partition_document(file_path)
    '''2. Chunk creation'''
    chunks = create_chunks_by_title(elements)
    '''3. Summarize chunks '''
    processed = summarized_chunks(chunks)
    '''4. vector store'''
    db = vector_store(processed, persist_directory="dbv2/chroma_db")
    print(f"Ingestion pipeline completed")

    return db

db = complete_ingestion_pipeline(file_path)

## Query the vector store
query = "What are the two main components of the Transformer architecture?"

# Retrieve from vector store    
retriever = db.as_retriever(search_kwargs={'k': 3})

chunks = retriever.invoke()
export_chunks_to_json(chunks, "rag_results.json")

def generation_of_answer(chunks, query):
    '''Generate final answer using multi-modal content'''
    try

    

