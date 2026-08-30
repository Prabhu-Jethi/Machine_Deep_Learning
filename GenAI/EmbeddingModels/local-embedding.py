from langchain_huggingface import HuggingFaceEmbeddings

def local_embeddingmodel():
    embedding = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v5-text-small",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"task": "retrieval"},
    )
    return embedding

embedding = local_embeddingmodel()

texts = [
    "Hello I'm Peter Parker aka Spiderman",
    "People often called me as friendly neighbourhood spiderman",
    "At first I thought that I'm made for the big leagues but after following mr.stark's advice",
    "I believe that I should stay in the ground cause the new avengers are there for the big potato stuffs"
]

vector = embedding.embed_documents(texts)
print(vector)