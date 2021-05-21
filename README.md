Topic modelling
---------------

Classify text in a document to a particular topic. It builds a topic per document model and 
words per topic model modelled as Dirichlet distributions.

Latent Dirichlet Allocation
______________
Each document is modeled as a multinomial distribution of topics and each topic is modeled as a multinomial distribution of words.
LDA assumes that the every chunk of text we feed into it will contain words that are somehow related. Therefore choosing the right corpus of data is crucial.
It also assumes documents are produced from a mixture of topics. Those topics then generate words based on their probability distribution.



