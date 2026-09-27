from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever
from langchain_cohere import CohereRerank
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

chunks = [
    # Tesla - Financial & Production
    "Tesla reported record quarterly revenue of $25.2 billion in Q3 2024.",
    "Tesla's automotive gross margin improved to 19.3% this quarter.",
    "Tesla Cybertruck production ramp begins in 2024 with initial deliveries.",
    "Tesla announced plans to expand Gigafactory production capacity.",
    "Tesla stock price reached new highs following earnings announcement.",
    "Tesla's energy storage business grew 40% year-over-year.",
    "Tesla continues to lead in electric vehicle market share globally.",
    "Tesla Model Y became the best-selling vehicle worldwide.",
    "Tesla reported strong free cash flow generation of $7.5 billion.",
    "Tesla's Full Self-Driving revenue increased significantly.",
    
    # Microsoft - Development & Acquisitions
    "Microsoft acquired GitHub for $7.5 billion in 2018.",
    "Microsoft's cloud revenue Azure grew 29% year-over-year.",
    "Microsoft announced new AI features for Visual Studio Code.",
    "Microsoft Teams integration with GitHub enhances developer workflow.",
    "Microsoft's developer tools division sees strong adoption.",
    "Microsoft acquired Activision Blizzard for $68.7 billion.",
    "Microsoft's productivity suite gained 50 million new users.",
    "Microsoft announced new Surface devices for developers.",
    "Microsoft's AI Copilot features expand to more development tools.",
    "Microsoft's enterprise solutions drive revenue growth.",
    
    # NVIDIA - AI & Hardware
    "NVIDIA's data center revenue reached $47.5 billion annually.",
    "NVIDIA's H100 GPUs see unprecedented demand for AI training.",
    "NVIDIA announced next-generation Blackwell architecture.",
    "NVIDIA's gaming revenue declined due to crypto market changes.",
    "NVIDIA's automotive AI platform partnerships expanded.",
    "NVIDIA's AI chip shortage affects cloud providers.",
    "NVIDIA stock valuation exceeds $2 trillion market cap.",
    "NVIDIA's CUDA platform dominates AI development.",
    "NVIDIA announced new AI inference chips for edge computing.",
    "NVIDIA's partnership with major cloud providers strengthens.",
    
    # Google/Alphabet - AI & Cloud
    "Google's AI investments total over $100 billion in recent years.",
    "Google Cloud revenue grew 35% reaching $8.4 billion quarterly.",
    "Google announced Gemini AI model competing with GPT-4.",
    "Google's search advertising revenue remains strong at $59 billion.",
    "Google's Workspace products integrate advanced AI features.",
    "Google announced quantum computing breakthroughs.",
    "Google's autonomous vehicle division Waymo expands operations.",
    "Google's AI research published breakthrough papers.",
    "Google's cloud AI services see enterprise adoption.",
    "Google faces regulatory scrutiny over AI dominance.",
    
    # Noisy/Less Relevant Chunks
    "The Tesla coil was invented by Nikola Tesla in 1891.",
    "Microsoft Excel spreadsheet formulas can be complex for beginners.",
    "NVIDIA Shield TV streaming device gets software update.",
    "Google Maps navigation improved with real-time traffic data.",
    "Production delays affected multiple manufacturing sectors.",
    "Financial markets showed volatility during earnings season.",
    "Revenue recognition standards changed for software companies.",
    "Hardware components face supply chain constraints globally.",
    "Development tools market grows with remote work trends.",
    "AI research requires significant computational resources.",
    "Quarterly reports show mixed results across tech sector.",
    "Stock market analysts upgrade technology sector ratings.",
    "Cloud computing adoption accelerates in enterprise market.",
    "Data center construction increases globally.",
    "Semiconductor shortage impacts various industries.",
    "Electric vehicle charging infrastructure expands rapidly.",
    "Software development productivity tools gain popularity.",
    "Machine learning frameworks become more accessible.",
    "Enterprise software licensing models evolve.",
    "Technology conferences showcase latest innovations."
]

## Converting to langchain documents
documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]
for i, chunk in enumerate(chunks, 1):
    print(f"{i}. {chunk}")

##---------------------------------------
##  1. Vector Retrieval
##---------------------------------------
def vector_retrieval(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    vect_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    return vect_retriever


##----------------------------------------
##  2. BM25Retriever
##----------------------------------------
def bm25_retrieval(documents):
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = 15
    return bm25


##-----------------------------------------
##  3. Hybrid Retriever
##-----------------------------------------
def hybrid_retrieval(vect_retriever, bm25_retriever):
    print("\n Hybrid Retriever loading...")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vect_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    ## Testing queries to get hybrid results
    query = "Tesla financial performance and production updates"
    print("\nSTEP 1: Hybrid Search Results")
    print("-"*50)

    retrieved_docs = hybrid_retriever.invoke(query)  # Get top 25 for reranking

    # Show top 10 from hybrid search
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i:2d}. {doc.page_content}")

    ## Applying `Cohere Reranking`
    print("\nSTEP 2: Cohere ReRanking (top 10)")
    print("-"*50)

    ## Initialize re-ranker model 
    reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
    reranked_docs = reranker.compress_documents(retrieved_docs, query)

    # Show reranked results
    for i, doc in enumerate(reranked_docs, 1):
        print(f"{i:2d}. {doc.page_content}")
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("✅ Hybrid Search: Mixed relevant and irrelevant results")
    print("✅ Reranking: Most relevant Tesla financial/production info at top")
    print("✅ Notice how reranking moved the most contextually relevant chunks higher")

    hybrid_top_5 = [doc.page_content for doc in retrieved_docs[:5]]
    print("\nHybrid Top 5:")
    for i, content in enumerate(hybrid_top_5, 1):
        print(f"  {i}. {content}")

    reranked_top_5 = [doc.page_content for doc in reranked_docs[:5]]
    print("\nReRanking Top 5:")
    for i, content in enumerate(reranked_top_5, 1):
        print(f"  {i}. {content}")
    return query, reranked_docs


def combined_documents(query, chunks):
    ## Using Top `n` reranked document to get final answer 
    top_reranked = chunks[:5]

    combined_inputs = f"""Based on the following documents, please answer this question: {query}
    Documents: {chr(10).join([f"- {doc.page_content}" for doc in top_reranked])}
    Please provide a clear, helpful answer using only the information from these documents."""
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    messages = [
        SystemMessage(content="You are a helpful assistance"),
        HumanMessage(content=combined_inputs)
    ]
    result = llm.invoke(messages)
    return result.content


def main():
    vector = vector_retrieval(documents)
    bm25 = bm25_retrieval(documents)
    query, chunks = hybrid_retrieval(vector, bm25)
    result = combined_documents(query, chunks)
    print(f"{result}")


if __name__ == "__main__":
    main()