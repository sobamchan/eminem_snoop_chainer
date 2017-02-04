from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import numpy as np
import json

def get_datas(filepath):
    with open(filepath, 'r') as f:
        datas = json.load(f)
    lyrics = []
    for data in datas:
        lyrics.append(' '.join(data['lyrics']).replace('\n', ''))

    return lyrics

def get_vectorizer(all_text):
    vectorizer = CountVectorizer()
    vectorizer.fit(all_text)
    return vectorizer

if __name__ == '__main__':
    r = get_datas('./datas/eminem.json')

def accuracy(Y, T):
    correct = 0
    incorrect = 0
    for y, t in zip(Y, T):
        if np.argmax(y) == t:
            correct += 1
        else:
            incorrect += 1
    return correct / (correct + incorrect) * 100
