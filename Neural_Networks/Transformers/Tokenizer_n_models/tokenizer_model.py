from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

'''The AutoModel class is a convenient way to load an architecture without needing to know the exact model class name because there are many
models available. It automatically selects the correct model class based on the configuration file. You only need to know the task and checkpoint 
you want to use.'''

model = AutoModelForSequenceClassification.from_pretrained(model_name)

''' -> A tokenizer converts text into tensors, which are the inputs to a model. 
It normalizes and splits text, applies the tokenization algorithm, adds special tokens, and decodes output ids back into text.
 -> AutoTokenizer.from_pretrained() reads the model config, resolves the correct tokenizer class, and returns an instance of it.'''

tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
res = classifier("I've been waiting for this moment so desperately")
print(res)