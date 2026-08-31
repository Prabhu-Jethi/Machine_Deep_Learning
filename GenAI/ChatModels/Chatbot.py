import os
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

def chatbot_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        model_kwargs={
            "local_files_only": True,
        },
        pipeline_kwargs=dict(
            max_new_tokens=256,
            temperature=0.9
        ),
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model


def main():
    print("Loading TinyLlama chatbot...")
    chat_model = chatbot_model()
    print("Chatbot ready! Type 'quit' to exit.\n")

    # Conversation history
    chat_history = [
        SystemMessage(content="You are a helpful assistant. Keep your answers short and clear."),
    ]

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # Add user message to history
        chat_history.append(HumanMessage(content=user_input))

        # Get response from model
        response = chat_model.invoke(chat_history)

        # Add AI response to history
        chat_history.append(AIMessage(content=response.content))

        print(f"Bot: {response.content}\n")


if __name__ == "__main__":
    main()