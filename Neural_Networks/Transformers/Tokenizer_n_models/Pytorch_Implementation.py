from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

X_train = ["I've been waiting for a Huggingface course my whole life.",
           "Data Science is great."]

res = classifier(X_train)
print("\nPipeline-result:", res)


batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
print("\nBatch-Processing:", batch)


with torch.no_grad():
    ## ** batch -> Unpacked batch
    outputs = model(**batch)
    print("\nModel outputs:", outputs)
    ## Applying softmax 
    predictions = F.softmax(
        outputs.logits,
        dim=1
    )
    print("\nPredictions:", predictions)
    ## Labels
    labels = torch.argmax(
        predictions,
        dim=1
    )
    print("\nLabels:", labels)

### Save Model and Tokens
save_directory = "saved"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

tok = AutoTokenizer.from_pretrained(save_directory)
mod = AutoModelForSequenceClassification.from_pretrained(save_directory)