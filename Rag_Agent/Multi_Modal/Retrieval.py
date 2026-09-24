import os
import json
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from Ingestion import export_chunks_to_json

# Optional: If you haven't run Ingestion.py yet, you can import and run it here.
# However, you should generally run Ingestion.py ONCE, and then just read the database here!
# from Ingestion import partition_document, create_chunks_by_title, summarized_chunks

load_dotenv()

file_path = "../docs/attention-is-all-you-need.pdf"

def generation_of_answer(chunks, query):
    '''Generate final answer using multi-modal content hidden in the chunk metadata'''
    try:
        llm = ChatOpenAI(
            model="gpt-5.6-luna", 
            base_url="https://api.experientiallabs.ai/v1", 
            api_key=os.environ["EXPLABS_API_KEY"]
        )
        
        # 1. Extract information from the retrieved chunks
        text_context = ""
        table_context = ""
        image_list = []
        
        for i, chunk in enumerate(chunks):
            # Parse the JSON we stored during ingestion
            data = json.loads(chunk.metadata["original_content"])
            # add raw text
            text_context += f"Chunk {i+1} Text:\n{data.get('raw_text', '')}\n\n"
            # add tables as html
            if data.get('tables_html'):
                for table in data['tables_html']:
                    table_context += f"{table}\n"
                    
            # If there are images, add them to our list
            if data.get('images_base64'):
                image_list.extend(data['images_base64'])

        # 2. Build the text prompt
        prompt = f"""Based on the following documents, please answer this question: {query}
        
        TEXT CONTENT:
        {text_context}
        
        TABLES:
        {table_context}
        """

        # 3. Assemble the multi-modal message list
        message_content = [{"type": "text", "text": prompt}]
        
        for img_base64 in image_list:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })

        # 4. Generate the answer!
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content
        
    except Exception as e:
        return f"Error generating answer: {e}"


def main():
    # CONNECT TO EXISTING VECTOR DB 
    print("Connecting to Vector Database...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Notice we use Chroma(...) to read an existing DB, rather than Chroma.from_documents(...)
    db = Chroma(
        persist_directory="dbv1/chroma_db",
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    query = "According to Table 1, what are the main advantages of self-attention layers compared to recurrent and convolutional layers in terms of computational complexity and parallelization? "
    print(f"\nSearching for: '{query}'")
    
    # Retrieve from vector store    
    retriever = db.as_retriever(search_kwargs={'k': 5})
    chunks = retriever.invoke(query)
    export_chunks_to_json(chunks, "rag_results.json")

    # Generate Answer
    answer = generation_of_answer(chunks, query)
    
    print("\n Final Answer Generated...")
    print(f"\n{answer}\n")


if __name__ == "__main__":
    main()

