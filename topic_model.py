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


