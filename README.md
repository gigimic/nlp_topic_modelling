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

