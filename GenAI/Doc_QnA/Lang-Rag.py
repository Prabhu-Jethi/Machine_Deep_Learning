from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


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
    
    ### 3. Embedding using Sentence-transformers, vectorstore or Chroma and retriever
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
    
    ### 4. Loading local LLM from HuggingFace
    def create_llm():
        llm = HuggingFacePipeline.from_model_id(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            task="text-generation",
            pipeline_kwargs=dict(
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.03,
            ),
        )
        return llm
    
    ### 5. Prompt-Template
    def create_prompt():
        system_prompt = (
            "<|system|>\n"
            "You are a helpful assistant. Use the following context to answer the question. "
            "If you don't know the answer, just say that you don't know.\n"
            "Context: {context}</s>\n"
            "<|user|>\n"
            "{input}</s>\n"
            "<|assistant|>\n"
        )
        prompt = ChatPromptTemplate.from_template(system_prompt)
        return prompt
    
    ### 6. Creating RAG-Chain pipeline
    def create_rag_chain(retriever, prompt, llm):
        stuff_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, stuff_chain)
        return rag_chain
    
    
    # --- Run the pipeline ---
    docs = load_doc(r"C:\Users\sudip\Downloads\cv.pdf")
    chunks = chunk_docs(docs)
    retriever = create_vectorstore_retriever(chunks)
    local_llm = create_llm()
    prompt = create_prompt()
    rag_chain = create_rag_chain(retriever, local_llm, prompt)


    print(f"Loaded {len(docs)} pages")
    print(f"Split into {len(chunks)} chunks")
    print("Rag pipeline created!")
    
    questions = "What are the skills?"
    response = rag_chain.invoke({'inputs': questions})
    print(f"Q: {questions}")
    print(f"A: {response['answer']}")



if __name__ == "__main__":
    build_local_rag()

