import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

if "HF_TOKEN" not in os.environ and "HUGGINGFACEHUB_API_TOKEN" in os.environ:
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACEHUB_API_TOKEN"]

from huggingface_hub import InferenceClient

# Use HF's free serverless Inference API
client = InferenceClient(token=os.environ.get("HF_TOKEN"))

# List a few supported models to find one that works
response = client.chat_completion(
    model="google/gemma-2-2b-it",
    messages=[{"role": "user", "content": "what is data science?"}],
    max_tokens=512,
)

print(response.choices[0].message.content)
