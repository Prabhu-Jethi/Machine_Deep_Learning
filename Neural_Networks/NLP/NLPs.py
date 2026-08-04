
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


### Dataset
df = pd.read_csv(r"C:\Users\sudip\Downloads\train.txt", sep=';', header=None, names=['text', 'emotion'])
print("\nSentiment-Analysis:\n", df)

print("\n", df.head())
print("\n", df.tail())

### Cleaning
print("\nNull_Check:\n", df.isnull().sum())


## convert emotions into numeric values
unique_emotions = df['emotion'].unique()
print("\nUnique Emotions:\n", unique_emotions)

emotion_numbers = {}
i = 0
for emo in unique_emotions:
    emotion_numbers[emo] = i
    i += 1
df['emotion'] = df['emotion'].map(emotion_numbers)
print("\nEmotion_Numbers:\n", emotion_numbers)


### Lowercasing
df['text'] = df['text'].apply(lambda x: x.lower())
print("\nLowerCase-Texts\n", df['text'])

import string
def remove_punc(txt):
    return txt.translate(str.maketrans('','', string.punctuation))

df['text'] = df['text'].apply(remove_punc)
print("\nPunctuations_Removed:\n", df['text'])

def remove_num(txt):
    new=''
    for i in txt:
        if not i.isdigit():
            new += i
    return new

df['text'] = df['text'].apply(remove_num)
print("\nNumbers_Removed:\n", df['text'])


### Stopword removal
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
print("\nStopWords:\n", stop_words)

lemmatizer = WordNetLemmatizer()
print("\nLemmatizer:\n", lemmatizer)

def preprocess(text):
    text = remove_punc(text)
    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return ' '.join(words)

df['text'] = df['text'].apply(preprocess)
print("\nLemmatized_texts:\n", df['text'])

### Separting for training
X = df['text']
y = df['emotion']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_val.shape)

### Convert using Box of words or Count vectorizer
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_val_bow = bow_vectorizer.transform(X_val)
print("\nBow_X_train:\n", X_train_bow)
print("\nBow_X_val:\n", X_val_bow)

model = MultinomialNB()
print("\n", model.fit(X_train_bow, y_train))

pred_bow = model.predict(X_val_bow)
print("\nAcc", accuracy_score(y_val, pred_bow))

### Convert Texts into Vectors (using TF-IDF)
Tfid_vectorizer = TfidfVectorizer()
X_train_Tfid = Tfid_vectorizer.fit_transform(X_train)
X_val_Tfid = Tfid_vectorizer.transform(X_val)
print("\nTfid_X_train_Vectorized:\n", X_train_Tfid)
print("\nTfid_X_val_Vectorized:\n", X_val_Tfid)


model = MultinomialNB()
print("\n", model.fit(X_train_Tfid, y_train))

pred_Tfid = model.predict(X_val_Tfid)
print("\nAccuracy_Score:\n", accuracy_score(y_val, pred_Tfid))

