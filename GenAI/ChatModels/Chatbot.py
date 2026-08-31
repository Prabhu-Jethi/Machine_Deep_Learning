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
            "low_cpu_mem_usage": True,
        },
        pipeline_kwargs=dict(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.9,
            repetition_penalty=1.03,
        ),
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model


def main():
    print("Loading TinyLlama chatbot...")
    try:
        chat_model = chatbot_model()
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    print("Chatbot ready! Type 'quit' to exit.\n")

    choice = int(input("Tell Your Response..."))

    for i in range(1, 4):
        if choice == 1:
            mode = "You are an angry AI agent. You respond aggressively and impatiently"
        elif choice == 2:
            mode = "You are a very funny AI agent. You respond with humor and jokes."
        elif choice == 3:
            mode = "You are a very sad AI agent. You respond in a depressed and emotional tone."
        else:
            mode = "You are a normal AI agent. You respond in a more theoritically and practically."

    # Conversation history
    chat_history = [
        SystemMessage(content=mode),
    ]

    while True:
        user_input = input("You: ")
        print("...")
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