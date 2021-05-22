import pandas as pd

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)

import nltk
# nltk.download('wordnet')

data = pd.read_csv('abcnews_date_text.csv', error_bad_lines=False);
# We only need the Headlines text column from the data

print(data.head())
print(len(data))
data_text = data[:30000][['headline_text']];
data_text['index'] = data_text.index

documents = data_text
print(len(documents))
print(documents[:5])

# preprocessing data

'''
Write a function to perform the pre processing steps on the entire dataset
'''

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

stemmer = SnowballStemmer("english")
# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            # TODO: Apply lemmatize_stemming on the token, then add to the results list
            result.append(lemmatize_stemming(token))
    return result


# Preview a document after preprocessing

document_num = 101
doc_sample = documents[documents['index'] == document_num].values[0][0]

print("Original document: ")
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print("\n\nTokenized and lemmatized document: ")
print(preprocess(doc_sample))