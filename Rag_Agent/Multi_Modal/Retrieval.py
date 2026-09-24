import os
from langchain_chroma import Chroma
from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from Ingestion import partition_document, create_chunks_by_title, summarized_chunks, vector_store, export_chunks_to_json
from dotenv import load_dotenv

load_dotenv()

file_path = "../docs/attention-is-all-you-need.pdf"

## Summarized chunks
summarized = summarized_chunks(chunks=create_chunks_by_title)
def re_initialize_vector_db(documents, persist_directory="dbv2/chroma_db"):
    ## Embedding and vector store
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        documents=documents,
        persist_directory=persist_directory,
        embedding=embedding_model,
        collection_metadata={"hnsw:space", "cosine"}
    )

    return vector_db

vector_db = re_initialize_vector_db(documents=)
## Query the vector store
query = "What are the two main components of the Transformer architecture?"

# Retrieve from vector store    
retriever = vector_db.as_retriever(search_kwargs={'k': 3})

chunks = retriever.invoke(query)
export_chunks_to_json(chunks, "rag_results.json")

def generation_of_answer(chunks, query):
    '''Generate final answer using multi-modal content'''
    try:
        llm = OpenAI(model="gpt-5.6-luna", base_url="https://api.experientiallabs.ai/v1", api_key=os.environ["EXPLABS_API_KEY"])
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""     

    

