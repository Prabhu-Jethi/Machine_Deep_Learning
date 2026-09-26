from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

# set up
text = "../docs/Tesla.txt"
persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

loader = TextLoader(text, encoding="utf-8")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = splitter.split_documents(documents)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persistent_directory,
    collection_metadata={"hnsw:space": "cosine"}
)

## Pydantic Model --> For Structured output
class QueryValidations(BaseModel):
    queries: List[str]


'''Original Query'''
org_query = "How does Tesla make money ?"
print(f"\nOriginal query: {org_query}")


''' Step 1. Generate multiple query validations '''

llm_with_tools = llm.with_structured_output(QueryValidations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:
Original query: {org_query}
Return 3 alternative queries that rephrase or approach the same question from different angles.
You MUST use the provided tool to output the response. Do not output any conversational text."""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"\n{i}. {variation}")


''' Step 2. Search with each query variation and stored results'''

retriever = db.as_retriever(search_kwargs={"k": 3})
all_retrieval_results = [] # Store all results for RRF

for i, query in enumerate(query_variations, 1):
    print(f"\nRESULTS FOR QUERY {i}: {query}")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)  # Store for RRF calculation
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
