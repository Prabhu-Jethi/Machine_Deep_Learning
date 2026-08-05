
from transformers import AutoTokenizer

'''The TokenizersBackend.call() method encodes text or a batch of text into input_ids, attention_mask, and other model inputs. 
It also controls padding, truncation, and special token insertion'''

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

'''TokenizersBackend.encode() is similar but only returns the input_ids.'''
### Converts sentence to words then to tokens
sequence = 'Using a transformer network is simple'
res = tokenizer(sequence)
print(res)

### Words
tokens = tokenizer.tokenize(sequence)
print("\nWords:", tokens)

'''TokenizersBackend.decode() converts a single sequence or batch of tokenized input_ids back to text.'''
### Integer ids of the words
ids = tokenizer.convert_tokens_to_ids(tokens)
print("\nTokenized_ids:", ids)

### Decoded ids converted to words
decoded_string = tokenizer.decode(ids)
print("\nDecoded_Words:", decoded_string)