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

# document_num = 1001
# doc_sample = documents[documents['index'] == document_num].values[0][0]

# print("Original document: ")
# words = []
# for word in doc_sample.split(' '):
#     words.append(word)
# print(words)
# print("\n\nTokenized and lemmatized document: ")
# print(preprocess(doc_sample))

# Now  use the map function from pandas to apply preprocess() to the headline_text column
# so that we can preprocess all the headlines
processed_docs = documents['headline_text'].map(preprocess)

# print(processed_docs[:10])
'''
Create a dictionary from 'processed_docs' containing the number of times a word appears 
in the training set using gensim.corpora.Dictionary and call it 'dictionary'
'''
dictionary = gensim.corpora.Dictionary(processed_docs)

# checking the first 10 words in the dictionary 
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

print(len(dictionary))

'''
OPTIONAL STEP
Remove very rare and very common words:

- words appearing less than 15 times
- words appearing in more than 10% of all documents
'''
# TODO: apply dictionary.filter_extremes() with the parameters mentioned above
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)