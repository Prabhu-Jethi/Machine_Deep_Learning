import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

model = ChatGroq(model="openai/gpt-oss-120b")

## Store conversation messages
chat_history = []


def ask_question(user_question):
    print(f"\n You asked {user_question}")

    ## 1. Make question clear using conversation history
    if chat_history:
        ## Ask AI to make question stand alone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. " \
            "Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New Question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching up for: {search_question}")
    else:
        search_question = user_question

    ## 2. Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"Doc {i}: {preview}")
    
    ## 3. Create Prompt 
    combined_input = f"""Based on the following documents, please answer this question: {user_question}
    Documents: {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, 
    say "I don't have enough information to answer that question based on the provided documents."""

    ## 4. Get the answer
    messages = [
        SystemMessage(content="You are an helpful assistance that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    ## 5. Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer


def start_chat():
    print("Ask me a question! Type 'q' to exit")

    while True:
        question = input("\n")

        if question.lower() == 'quit':
            print("Thank you. Have a nice day")
            break

        ask_question(question)



if __name__ == "__main__":
    start_chat()

