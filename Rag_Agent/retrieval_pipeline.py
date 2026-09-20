import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

def retrieval_model(query):
    # 1. We MUST use the exact same embedding model used during ingestion!
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 2. Connect to the existing Chroma database
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # 3. Create our retriever and search for relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)
    
    context = ""
    for i, doc in enumerate(relevant_docs, 1):
        context += f"--- Document {i} ---\n{doc.page_content}\n\n"
        
    return context


def main():
    # 1. Get the user query
    query = "What was Microsoft's first hardware product release?"

    # 2. Get the context from our RAG database
    print("\nSearching database for relevant context...")
    context = retrieval_model(query)

    # 3. Pass the context and the question to Groq!
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": "What was Microsoft's first hardware product release?"
            }
        ]
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()




# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"