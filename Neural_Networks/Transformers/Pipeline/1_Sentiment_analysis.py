from transformers import pipeline

## Sentiment-Analysis
classifier = pipeline("sentiment-analysis")
result = classifier("The new Spiderman movie: Brand new day is amazing")
print(result)