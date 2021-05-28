Topic modelling
---------------

Classify text in a document to a particular topic. It builds a topic per document model and 
words per topic model modelled as Dirichlet distributions.

Latent Dirichlet Allocation
______________
Each document is modeled as a multinomial distribution of topics and each topic is modeled as a multinomial distribution of words.
LDA assumes that the every chunk of text we feed into it will contain words that are somehow related. Therefore choosing the right corpus of data is crucial.
It also assumes documents are produced from a mixture of topics. Those topics then generate words based on their probability distribution.

Data Preprocessing
------------------

Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
Words that have fewer than 3 characters are removed.
All stopwords are removed.
Words are lemmatized - words in third person are changed to first person and verbs in past and future tenses are changed into present.
Words are stemmed - words are reduced to their root form.

Bag of words on the dataset
---------------------------

We create a dictionary from 'processed_docs' containing the number of times a word appears in the training set. To do that, let's pass processed_docs to gensim.corpora.Dictionary() and call it 'dictionary'.

Gensim filter_extremes
---------------------
Filter out tokens that appear in

less than no_below documents (absolute number) or
more than no_above documents (fraction of total corpus size, not absolute number).
after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).

Gensim doc2bow
-----------------

doc2bow(document)

Convert document (a list of words) into the bag-of-words format = list of (token_id, token_count) 2-tuples. Each word is assumed to be a tokenized and normalized string (either unicode or utf8-encoded). No further preprocessing is done on the words in document; apply tokenization, stemming etc. before calling this method.

TF-IDF ("Term Frequency, Inverse Document Frequency") on our document set
------------------------------
While performing TF-IDF on the corpus is not necessary for LDA implemention using the gensim model, it is recemmended. TF-IDF expects a bag-of-words (integer values) training corpus during initialization. During transformation, it will take a vector and return another vector of the same dimensionality.

Please note: The author of Gensim dictates the standard procedure for LDA to be using the Bag of Words model.

It is a way to score the importance of words (or "terms") in a document based on how frequently they appear across multiple documents.
If a word appears frequently in a document, it's important. Give the word a high score. But if a word appears in many documents, it's not a unique identifier. Give the word a low score.
Therefore, common words like "the" and "for", which appear in many documents, will be scaled down. Words that appear frequently in a single document will be scaled up.

Running LDA using Bag of Words
-----------------------------

We are going for 10 topics in the document corpus.

We will be running LDA using all CPU cores to parallelize and speed up model training.

Some of the parameters we will be tweaking are:

num_topics: is the number of requested latent topics to be extracted from the training corpus.
id2word:  is a mapping from word ids (integers) to words (strings). It is used to determine the vocabulary size, as well as for debugging and topic printing.
workers: is the number of extra processes to use for parallelization. Uses all available cores by default.
alpha and eta: are hyperparameters that affect sparsity of the document-topic (theta) and topic-word (lambda) distributions. We will let these be the default values for now(default value is 1/num_topics)

Alpha:  is the per document topic distribution.

High alpha: Every document has a mixture of all topics(documents appear similar to each other).
Low alpha: Every document has a mixture of very few topics

Eta is the per topic word distribution.

High eta: Each topic has a mixture of most words(topics appear similar to each other).
Low eta: Each topic has a mixture of few words.
passes:  is the number of training passes through the corpus. For example, if the training corpus has 50,000 documents, chunksize is 10,000, passes is 2, then online training is done in 10 updates:

Running LDA using TF-IDF
-----------------------

The same can be done with TF-IDF instead of BOW

The model can be tested with a new document to find the efficiency of the model.