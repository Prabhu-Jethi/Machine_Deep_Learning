from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

## Sample data
chunks = [
    "Microsoft acquired GitHub for 7.5 billion dollars in 2018.",
    "Tesla Cybertruck production ramp begins in 2024.",
    "Google is a large technology company with global operations.",
    "Tesla reported strong quarterly results. Tesla continues to lead in electric vehicles. Tesla announced new manufacturing facilities.",
    "SpaceX develops Starship rockets for Mars missions.",
    "The tech giant acquired the code repository platform for software development.",
    "NVIDIA designs Starship architecture for their new GPUs.",
    "Tesla Tesla Tesla financial quarterly results improved significantly.",
    "Cybertruck reservations exceeded company expectations.",
    "Microsoft is a large technology company with global operations.", 
    "Apple announced new iPhone features for developers.",
    "The apple orchard harvest was excellent this year.",
    "Python programming language is widely used in AI.",
    "The python snake can grow up to 20 feet long.",
    "Java coffee beans are imported from Indonesia.", 
    "Java programming requires understanding of object-oriented concepts.",
    "Orange juice sales increased during winter months.",
    "Orange County reported new housing developments."
]

## Convert to langchain documents
documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]

print("\nSample Data...")
for i, chunk in enumerate(chunks, 1):
    print(f"{i}. {chunk}")

##---------------------------------------------------------------
##  1. Vector Retriever (Semantic Search / Dense Retrieval)
##---------------------------------------------------------------
def vector_retriever(documents):
    print("\n Vector Retriever...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    # test semantic search
    test_query = "space exploration company"    ## Only works with vector search and won't work in case of keyword search
    print(f"\nTest Query: {test_query}")
    test_docs = retriever.invoke(test_query)
    for doc in test_docs:
        print(f"\nFound: {doc.page_content}")
    return retriever

##--------------------------------------------------------
##  2. BM25 Retriever (Keyword search / Sparse retrieval)
##--------------------------------------------------------
def bm25_retriever(documents):
    print("\n Setting up BM25-Retriever...")
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = 3

    ## testing queries
    test_query = "Cybertruck"
    print(f"\nTesting: {test_query}")
    test_docs = bm25.invoke(test_query)
    for doc in test_docs:
        print(f"\n Found: {doc.page_content}")
    return bm25

##--------------------------------------------------------
##  3. Hybrid Retriever (Combination)
##--------------------------------------------------------
def hybrid_retriever(vector_ret, bm25_ret):
    print("\n Setting up Hybrid retriever...")
    hybrid = EnsembleRetriever(
        retrievers=[vector_ret, bm25_ret],
        weights=[0.7, 0.3]  # equal weight to vector and keyword search
    )
    ## testing
    '''1. Mixed Semantic and Exact terms'''
    test_query = "purchase cost 7.5 billion"
    retrieved_chunks = hybrid.invoke(test_query)
    for i, doc in enumerate(retrieved_chunks, 1):
        print(f"{i}. {doc.page_content}")
    print("\nQuery 1 shows how hybrid finds exact financial info using both semantic understanding and keyword matching")

    '''2. Semantic concept + specific product name'''
    test_query = "electric vehicle manufacturing Cybertruck"
    retrieved_chunks = hybrid.invoke(test_query)
    for i, doc in enumerate(retrieved_chunks, 1):
        print(f"{i}. {doc.page_content}")
    print("\nQuery 2 demonstates combining product-specific terms with broader concepts")

    '''3. Where neither would be perfect'''
    test_query = "electric vehicle manufacturing Cybertruck"
    retrieved_chunks = hybrid.invoke(test_query)
    for i, doc in enumerate(retrieved_chunks, 1):
        print(f"{i}. {doc.page_content}")
    print("\nQuery 3 shows how hybrid handles mixed semantic/keyword queries better than either approach alone")
    return test_query, retrieved_chunks



def combined_documents(query, chunks):
    combined_inputs = f"""Based on the following documents, answer these question: {query}
    Documents: {chr(10).join([f"- {doc.page_content}" for doc in chunks])} 
    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents,
    say "I don't have enough information to answer that question based on the provided documents."""

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content=combined_inputs)
    ]

    result = llm.invoke(messages)
    print("\n Generated Response...")
    print("\n Content only..")
    return result.content


def main():
    v_retriever = vector_retriever(documents)
    b_retriever = bm25_retriever(documents)
    query, chunks = hybrid_retriever(v_retriever, b_retriever)
    result = combined_documents(query, chunks)
    print(f"{result}")
    

if __name__ == "__main__":
    main()