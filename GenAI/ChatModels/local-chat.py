import os
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

def local_chatmodel():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task='text-generation',
        model_kwargs={
            "local_files_only": True,
        },
        pipeline_kwargs=dict(
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.03
        ),
    )
    return llm

llm = local_chatmodel()

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke("What is Langchain?")
print(result.content)