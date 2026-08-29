import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq


def build_local_rag():

    ### 1 and 2. Loading and Chunking documents
    def load_doc(path):
        loader = PyPDFLoader(path)
        docs = loader.load()
        return docs

    def chunk_docs(docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(docs)
        return chunks

    ### 3. Embedding using Sentence-transformers, vectorstore and retriever
    def create_vectorstore_retriever(chunks):
        hf_embed = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=hf_embed
        )
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        return retriever

    ### 4. LLM (using Groq - free & fast, runs remotely)
    def create_llm():
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
        )
        return llm

    ### 5. Prompt template
    def create_prompt():
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the provided context.
If you cannot find the answer in the context, say "I don't know".

Context: {context}

Question: {input}

Answer:"""
        )
        return prompt

    ### 6. Build the RAG chain
    def create_rag_chain(retriever, llm, prompt):
        stuff_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, stuff_chain)
        return rag_chain

    # --- Run the pipeline ---
    docs = load_doc(r"C:\Users\sudip\Downloads\cv.pdf")
    chunks = chunk_docs(docs)
    retriever = create_vectorstore_retriever(chunks)
    llm = create_llm()
    prompt = create_prompt()
    rag_chain = create_rag_chain(retriever, llm, prompt)

    print(f"Loaded {len(docs)} pages")
    print(f"Split into {len(chunks)} chunks")
    print("RAG pipeline ready!\n")

    ### 7. Query it!
    question = "What are the skills?"
    response = rag_chain.invoke({"input": question})
    print(f"Q: {question}")
    print(f"A: {response['answer']}")


if __name__ == "__main__":
    build_local_rag()
