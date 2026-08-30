from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#### OPEN_AI_API KEY required
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-large',
    dimensions=64
)

texts = [
    "Hello I'm Peter Parker aka Spiderman",
    "People often called me as friendly neighbourhood spiderman",
    "At first I thought that I'm made for the big leagues but after following mr.stark's advice",
    "I believe that I should stay in the ground cause the new avengers are there for the big potato stuffs"
]

vector = embeddings.embed_documents(texts)
print(vector)