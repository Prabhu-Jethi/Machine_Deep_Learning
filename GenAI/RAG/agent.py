import requests
import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain_huggingface import ChatHuggingFace
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


DOCS_BASE = "https://docs.langchain.com"

DOC_PATHS = [
    "oss/python/langchain/agents",
    "oss/python/deepagents/rag",
    "oss/python/langchain/tools",
    "oss/python/langchain/models",
    "oss/python/deepagents/retrieval",
    "oss/python/langchain/knowledge-base",
    "oss/python/langchain/middleware",
    "oss/python/deepagents/overview",
    "oss/python/deepagents/subagents",
    "oss/python/deepagents/streaming",
    "oss/python/deepagents/frontend/subagent-streaming",
    "oss/python/deepagents/backends",
    "oss/python/langgraph/overview",
    "oss/python/langgraph/quickstart",
]

## Loading
def load_langchain_docs():
    paths = DOC_PATHS
    docs = []
    for path in paths:
        url = f"{DOCS_BASE}/{path}.md"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status
        except requests.RequestException:
            continue
        source = f"{DOCS_BASE}/{path}"
        docs.append(
            Document(page_content=response.text, metadata={"source": source})
        )
    return docs

docs = load_langchain_docs()
print(f"Loaded {len(docs)} documentation pages.")


## Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
print(f"Split documentation into {len(splits)} chunks.")


## Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
## Store chunks in vectors
vector_store = Chroma(
    collection_name="vector_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)
vector_store.add_documents(documents=splits)
print(f"Indexed {len(splits)} chunks")


##### Building the agent

##------ 1. Add Search Tool
# Instead of writing a complex custom tool that saves files to a virtual filesystem,
# we can use LangChain's built-in retriever tool which does everything automatically!

search_documentation = create_retriever_tool(
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    name="search_documentation",
    description="Search LangChain documentation. Use this to find information about LangChain, agents, tools, and RAG."
)
tools = [search_documentation]


##------ 2. Load the Local LLM (TinyLlama)
def local_model():  
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        model_kwargs={
            "local_files_only": True,
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
        },
        pipeline_kwargs=dict(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1, # Low temp for agents to keep them focused
            repetition_penalty=1.03,
        ),
    )
    model = ChatHuggingFace(llm=llm)
    return model


print("Loading TinyLlama model...")
try:
    model = local_model()
except Exception as e:
    print(f"Error, Loading the model: {e}")


##------ 3. Create the Agent
# Standard ReAct prompt telling the model how to use tools
template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# Create the agent
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


##------ 4. Run agent
EXAMPLE_QUERY = "How do I stream intermediate tool results from a subagent?"

if __name__ == "__main__":
    print("\n=== Agent Ready ===")
    print(f"Question: {EXAMPLE_QUERY}\n")
    
    # Since we are using a classic ReAct agent (because TinyLlama doesn't support LangGraph tool calling),
    # we pass the text of the HumanMessage as the input.
    human_msg = HumanMessage(content=EXAMPLE_QUERY)
    
    result = agent_executor.invoke({"input": human_msg.content})
    
    print("\n=== Final Output ===")
    print(result['output'])