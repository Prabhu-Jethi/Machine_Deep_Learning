from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

text_path = "../docs/Microsoft.txt"
persistent_directory="db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

loader = TextLoader(text_path, encoding="utf-8")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)

db = Chroma.from_documents(
    documents=chunks,
    persist_directory=persistent_directory,
    embedding=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "How much did Microsoft pay to acquire GitHub?"
print(f"\nQuery: {query}\n")

'''Method 1: Basic Similarity Search -> Returns top k similarities'''
# retriever = db.as_retriever(search_kwargs={"k": 3})
# docs = retriever.invoke(query)
# print(f"Retrieved documents {(len(docs))}")

# for i, doc in enumerate(docs, 1):
#     print(f"Document {i}\n")
#     print(f"{doc.page_content}\n")

'''Method 2: Similarity with score threshold'''
# retreiver = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 3,
#         "score_threshold": 0.3
#     }
# )
# docs = retreiver.invoke(query)
# print(f"Retrieved documents {(len(docs))}")

# for i, doc in enumerate(docs, 1):
#     print(f"Document {i}\n")
#     print(f"{doc.page_content}\n")

'''Method 3: Maximum Marginal Relevance (MMR)'''
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,         # Final number of docs
        "fetch_k": 30,  # Initial pool to select from
        "lambda_mult": 0.5  # 0 -> Max diversity, 1 -> Max relevance
    }
)
docs = retriever.invoke(query)
print(f"Retrieved documents {(len(docs))}")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}\n")
    print(f"{doc.page_content}\n")