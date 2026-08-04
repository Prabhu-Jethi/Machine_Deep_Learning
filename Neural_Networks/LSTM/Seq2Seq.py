import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



''' Simple Translation of some words '''
#### <start> -> Beginning of sentence <end> -> Ending of sentence
english = [
    'hello',
    'how are you',
    'good morning',
    'thank you',
    'i am from india'
]
french = [
    '<start> bonjour <end>',
    '<start> comment allez vous <end>',
    '<start> bon matin <end>',
    '<start> merci <end>',
    '<start> j aime inde <end>'
]


''' Tokenization '''
### Encoder
eng_token = Tokenizer(filters="")
eng_token.fit_on_texts(english)
encoder_sequence = eng_token.texts_to_sequences(english)
encoder_vocab = len(eng_token.word_index) + 1
print("\nEncoder_Vocab:", encoder_vocab)

### Decoder
fra_token = Tokenizer(filters="<>")
fra_token.fit_on_texts(french)
decoder_sequence = fra_token.texts_to_sequences(french)
decoder_vocab = len(fra_token.word_index) + 1
print("\nDecoder_Vocab:", decoder_vocab)


#### Padding
'''English

hello
↓
[1 0 0]

how are you
↓
[2 3 4]'''
max_encoder_len = max(len(seq) for seq in encoder_sequence)
max_decoder_len = max(len(seq) for seq in decoder_sequence)

encoder_input = pad_sequences(
    encoder_sequence,
    maxlen=max_encoder_len,
    padding='post'
)
print("\nEncoder_Input:\n", encoder_input)

decoder_input = pad_sequences(
    decoder_sequence,
    maxlen=max_decoder_len,
    padding='post'
)
print("\nDecoder_Input:\n", decoder_input)

#### Create decoder target
'''Decoder Input	        Decoder Target
<start> bonjour <end>	    bonjour <end>'''
### Shift by 1 timestep
decoder_target = np.zeros_like(decoder_input)
decoder_target[:, :-1] = decoder_input[:, 1:]
print("\nDecoder_Target:\n", decoder_target)


#### Encoder Model
enc_inputs = Input(
    shape=(None,)
)
    
enc_embed = Embedding(
    input_dim=encoder_vocab,
    output_dim=64
)(enc_inputs)

enc_lstm = LSTM(
    units=128,
    return_state=True
)

### State h -> Hidden State, State c -> Cell State
enc_outputs, state_h, state_c = enc_lstm(
    enc_embed
)
enc_states = [state_h, state_c]


#### Decoder Model
dec_inputs = Input(
    shape=(None,)
)

dec_embedding = Embedding(
    input_dim=decoder_vocab,
    output_dim=64
)
dec_embed = dec_embedding(dec_inputs)

dec_lstm = LSTM(
    units=128,
    return_sequences=True,
    return_state=True
)

dec_outputs, _, _ = dec_lstm(
    dec_embed,
    initial_state=enc_states
)


##### Dense Layer
dec_dense = Dense(
    decoder_vocab,
    activation='softmax'
)
dec_outputs = dec_dense(dec_outputs)



##### SEQ 2 SEQ Model
def seq2seq_model():

    model = Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=[dec_outputs]
    )
    model.compile(
        optimizer=Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    return model

model = seq2seq_model()

decoder_target = np.expand_dims(decoder_target, axis=-1)
print("\nDecoder Target:\n", decoder_target.shape)

#### Model Training
history = model.fit(
    [encoder_input, decoder_input],
    decoder_target,
    epochs=300,
    batch_size=32,
    verbose=2
)
print(history.history["accuracy"][-1])
print(history.history["loss"][-1])

pred = model.predict([encoder_input, decoder_input], verbose=0)

print("\nPrediction shape:\n", pred.shape)

print("Prediction:")
print("\nPrediction:\n", np.argmax(pred[0], axis=-1))

print("\nTarget:\n", decoder_target[0].flatten())


print("\nEvaluate:\n", model.evaluate(
    [encoder_input, decoder_input],
    decoder_target,
    verbose=1
))

'''During training, the decoder receives the encoder's final states:
decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states
)
encoder_states = [state_h, state_c]

During inference, the encoder has already finished running.
The decoder must therefore accept the hidden and cell states as new inputs. 
That's why we create decoder_states_inputs.


Q- WHY DO WE NEED ENCODER INFERENCE MODEL ? 
A- The encoder converts an input sentence into hidden and cell states.

Q- WHY DO WE NEED DECODER INFERENCE MODEL ?
A-  Decoder predicts one word at a time

Previous Word
+
Hidden State
+
Cell State
        │
        ▼
Next Word

Q- WHY DO WE NEED NEW INPUTS() LAYERS ?

A- During training:

Encoder
   │
   ▼
Hidden State (h)
Cell State (c)
   │
   ▼
Decoder

The encoder directly supplies the states.

During inference:

Sentence
   │
   ▼
Encoder Model
   │
   ▼
(h, c)
   │
   ▼
Decoder Model
        ▲
        │
These states must be accepted as inputs.'''

''' Since the decoder is now a separate model, it has no connection to the encoder. 
Therefore, we create new Input() layers to receive the hidden and cell states at 
each decoding step.

This separation between the training model and the inference models is one of the most 
important concepts in Seq2Seq. Training uses teacher forcing with both encoder and
decoder together, while inference uses an encoder model once and then repeatedly 
calls the decoder model one token at a time until the <end> token is generated.'''


''' Encoder Inference Model '''
enc_model = Model(
    enc_inputs,
    enc_states
)

########## DECODER INFERENCE
##### Create state Input layers
decoder_state_input_h = Input(
    shape=(128,)
)
decoder_state_input_c = Input(
    shape=(128,)
)
decoder_state_inputs = [
    decoder_state_input_h,
    decoder_state_input_c
]

### Reuse the trained decoder embedding
decoder_embed2 = dec_embedding(dec_inputs)


### Reuse the trained decoder LSTM
dec_outputs, state_h, state_c = dec_lstm(
    decoder_embed2,
    initial_state=decoder_state_inputs
)

#### Reuse the trained Dense layer
dec_outputs = dec_dense(dec_outputs)

''' Decoder Inference Model '''
dec_model = Model(
    inputs=[dec_inputs] + decoder_state_inputs,
    outputs=[dec_outputs, state_h, state_c]
)



#### Reverse Tokenizer --> Convert predicted integer IDs back into word
reverse_target_word_index = {
    i: w
    for w, i in fra_token.word_index.items()
}
print("\nOriginal_token:\n", fra_token.word_index)
print("\nReversed_Target_word:\n", reverse_target_word_index)

### Prepare Input Sentence
def decode_sequence(input_seq):
    # Encode the input sentence
    states = enc_model.predict(input_seq, verbose=0)

    # Start token
    target_seq = np.array([[fra_token.word_index["start"]]])
    decoded_sentence = ""

    while True:
        ### Predict one word
        output_tokens, h, c = dec_model.predict(
            [target_seq] + states,
            verbose=1
        )
        print("\nOutput Tokens:\n", output_tokens[0, -1, :])

        ### Pick the best word
        sampled_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_index, "")

        print("\nVocabulary Size :", decoder_vocab)
        print("\nTarget Input:\n", target_seq)
        print("\nPredicted Index :", sampled_index)
        print("\nPredicted Word:\n", sampled_word)

        if sampled_word == "end" or sampled_word == "":
            break

        ### Save the word
        decoded_sentence += sampled_word + " "

        ### Feed the word back
        target_seq = np.array([[sampled_index]])
        states = [h, c]
        
        if len(decoded_sentence.split()) > max_decoder_len:
            break

    return decoded_sentence.strip()

#### Prepare Input senetence
sentence = 'how are you'

sequence = eng_token.texts_to_sequences([sentence])

sequence = pad_sequences(
    sequence,
    maxlen=max_encoder_len,
    padding='post'
)

translation = decode_sequence(sequence)
print("\nTranslated Word:", translation)