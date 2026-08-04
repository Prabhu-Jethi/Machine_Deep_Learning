from transformers import pipeline

classifier = pipeline(
    task="zero-shot-classification"
)

sequences="This is a course about Mutual Funds and SIP",
candidate_levels=["investments", "education", "politics"]

result = classifier(
    sequences, candidate_levels
)
print(result)


'''Device set to use cpu
{'sequence': 'This is a course about Mutual Funds and SIP', 
'labels': ['investments', 'education', 'politics'], 'scores': [0.84178227186203, 0.1573292464017868, 0.000888453156221658]}'''

