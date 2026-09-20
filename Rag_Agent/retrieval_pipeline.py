from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

def retrieval_model():
    client = Groq()
    embedding_model = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{
            "role": "user",
            "content": ""
        }]
    )

    db = Chroma(
        persistent_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    ## Search for your relevant documents
    query = ""

    retriever = db.as_retriever(search_kwargs={"k": 5})


    relevant_docs = retriever.invoke(query)

    print(f"User Query: {query}")
    # Display results
    print("--- Context ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")

    return retriever