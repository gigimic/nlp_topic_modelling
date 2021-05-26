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

document_num = 4310
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

print('length of the dictionary : ',len(dictionary))

'''
OPTIONAL STEP
Remove very rare and very common words:

- words appearing less than 15 times
- words appearing in more than 10% of all documents
'''
# TODO: apply dictionary.filter_extremes() with the parameters mentioned above
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)

'''
Gensim doc2bow

doc2bow(document)

Convert document (a list of words) into the bag-of-words format = list of (token_id, token_count) 2-tuples. 
Each word is assumed to be tokenized and normalized string (either unicode or utf8-encoded). 
No further preprocessing is done on the words in document; 
apply tokenization, stemming etc. before calling this method.
Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
words and how many times those words appear.

'''
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Checking Bag of Words corpus for our sample document --> (token_id, token_count)

print('bag of words for one sample document', bow_corpus[document_num])


# Preview BOW for our sample preprocessed document

# Here document_num is document number 4310 which we have checked in Step 2
bow_doc = bow_corpus[document_num]

for i in range(len(bow_doc)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc[i][0], 
                                                     dictionary[bow_doc[i][0]], 
                                                     bow_doc[i][1]))



# Create tf-idf model object using models.TfidfModel on 'bow_corpus' and save it to 'tfidf'

from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)

# Apply transformation to the entire corpus and call it 'corpus_tfidf'

corpus_tfidf = tfidf[bow_corpus]

# Preview TF-IDF scores for our first document --> --> (token_id, tfidf score)

from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

# running lda using bow 
# LDA mono-core
# lda_model = gensim.models.LdaModel(bow_corpus, 
#                                    num_topics = 10, 
#                                    id2word = dictionary,                                    
#                                    passes = 50)

# Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'

lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                       num_topics=5, 
                                       id2word = dictionary, 
                                       passes = 2, 
                                       workers=2)


# For each topic, we will explore the words occuring in that topic and its relative weight

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic))
    print("\n")

# Running LDA using TF-IDF

# Define lda model using corpus_tfidf

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, 
                                             num_topics=10, 
                                             id2word = dictionary, 
                                             passes = 2, 
                                             workers=4)


# For each topic, we will explore the words occuring in that topic and its relative weight

for idx, topic in lda_model_tfidf.print_topics(-1):
    print("Topic: {} Word: {}".format(idx, topic))
    print("\n")


