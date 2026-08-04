from transformers import pipeline

## Word Translation
translator = pipeline(
    'translation_en_to_fr',
    model="Helsinki-NLP/opus-mt-en-fr"
)
print(translator("I Love AI"))