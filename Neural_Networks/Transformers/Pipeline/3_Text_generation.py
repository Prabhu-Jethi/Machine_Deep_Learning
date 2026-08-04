
from transformers import pipeline

generator = pipeline(task="text-generation", model="distilgpt2")
result = generator(
    "the secret to baking a really good cake is ",
    num_return_sequences=2
)
print(result)