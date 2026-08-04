
from transformers import pipeline

pipe = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-large-v3"
)
res = pipe(
    inputs="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
)
print(res)


'''{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}'''
