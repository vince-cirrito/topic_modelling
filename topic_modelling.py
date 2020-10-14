# -*- coding: utf-8 -*-
"""
@author: cirrito
topic modelling

"""
import gensim
from gensim import matutils, models
import scipy.sparse
import src.feature_extraction
import pyLDAvis.gensim

def compute_LDA(text_documents, NUM_TOPICS, NUM_ITERATIONS, model_name):
    '''
    This function takes four argouments:
    1) the list of documents
    2) the number of topics
    3) iterations
    4) the name for saving the topic model
    
    '''
    #################### inputs for the gensim LDA model########################
    #corpus in a bag-of-words format
    text_documents = text_to_analyse
    bag_of_words = src.feature_extraction.get_count_vectorizer(text_documents, document_term = False)[0]
    sparse_counts = scipy.sparse.csr_matrix(bag_of_words)
    corpus = matutils.Sparse2Corpus(sparse_counts)

    #dictionary
    cv = src.feature_extraction.get_count_vectorizer(text_documents, document_term = False)[1]
    dictionary = dict((v, k) for k, v in cv.vocabulary_.items())
    
    # main algorithm
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=NUM_ITERATIONS)
    
    gensim_name = model_name + ".gensim"
    ldamodel.save(gensim_name)
    
    return [ldamodel, corpus, dictionary]

def print_LDA_topic(gensim_model_name, NUM_WORDS):
    ldamodel = gensim.models.ldamodel.LdaModel.load(gensim_model_name)
    topics = ldamodel.print_topics(num_words=NUM_WORDS)
    for topic in topics:
        print(topic)
    return 

"""
def get_visualization_lda_output:
    '''
    
    '''
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    vis
    
"""
df = src.feature_extraction.get_dataset()
text_to_analyse = list(df["finding_text"])

lda_model =compute_LDA(text_documents = text_to_analyse, NUM_TOPICS = 10, NUM_ITERATIONS = 5, model_name = 'experiment')

"""Visualize the groups """
ldamodel = gensim.models.ldamodel.LdaModel.load('experiment.gensim')

topics = ldamodel.print_topics(num_words=8)
for topic in topics:
    print(topic)
    
""" PyLDAvis """
vis = pyLDAvis.gensim.prepare(lda_model[0],corpus = corpus, dictionary = dictionary)
vis
