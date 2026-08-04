import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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


### Padding
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

decoder_target = np.zeros_like(decoder_input)
decoder_target[:, :-1] = decoder_input[:, 1:]
print("\nDecoder_Target:\n", decoder_target)


encoder_inputs = Input(shape=(None,))
encoder_embed = Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_inputs)
encoder_lstm = LSTM(units=128, return_state=True, return_sequences=True)

encoder_outputs, state_h, state_c = encoder_lstm(encoder_embed)


decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=decoder_vocab, output_dim=64)
## For inference call
decoder_embed = decoder_embedding(decoder_inputs)
decoder_lstm = LSTM(units=128, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(decoder_embed, initial_state=[state_h, state_c])

''' ATTENTION: Focuses only to the most relevant parts of inputs'''
# Decoder attends to encoder outputs
attention = Attention()([decoder_outputs, encoder_outputs])

decoder_dense = Dense(decoder_vocab, activation='softmax')
dense_output = decoder_dense(attention)

def Attention_model():
    model = Model(
    inputs=[encoder_inputs, decoder_inputs],
    outputs=dense_output
    )
    model.compile(
        optimizer=Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

model = Attention_model()


decoder_target = np.expand_dims(decoder_target, axis=-1)
print("\nDecoder Target:\n", decoder_target.shape)

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
for i in range(len(english)):
    print("\nEnglish:", english[i])
    print("Expected:", decoder_target[i].flatten())
    print("Predicted:", np.argmax(pred[i], axis=-1))
print("\nPrediction shape:\n", pred.shape)

print("\nPrediction:\n", np.argmax(pred[0], axis=-1))

print("\nTarget:\n", decoder_target[0].flatten())


print("\nEvaluate:\n", model.evaluate(
    [encoder_input, decoder_input],
    decoder_target,
    verbose=1
))

''' Encoder Inference Model '''
encoder_model = Model(
    encoder_inputs,
    [encoder_outputs, state_h, state_c]
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
decoder_reembed = decoder_embedding(decoder_inputs)


### Reuse the trained decoder LSTM
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_reembed,
    initial_state=decoder_state_inputs
)

#### Reuse the trained Dense layer
decoder_outputs = decoder_dense(decoder_outputs)

''' Decoder Inference Model '''
dec_model = Model(
    inputs=[decoder_inputs] + decoder_state_inputs,
    outputs=[decoder_outputs, state_h, state_c]
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
    states = encoder_model.predict(input_seq, verbose=0)

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
sentence = 'good morning'

sequence = eng_token.texts_to_sequences([sentence])

sequence = pad_sequences(
    sequence,
    maxlen=max_encoder_len,
    padding='post'
)

translation = decode_sequence(sequence)
print("\nTranslated Word:", translation)