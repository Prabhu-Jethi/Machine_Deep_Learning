
from transformers import pipeline

pipe = pipeline(
    task="question-answering",
    model="distilbert-base-cased-distilled-squad"
)
result = pipe(
    question="What is AI?",
    context="Artificial Intelligence is a branch of computer science. "
)
print(result)


'''{'score': 0.4557715356349945, 'start': 27, 'end': 55, 'answer': 'a branch of computer science'}'''