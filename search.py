import string
import math
from nltk import word_tokenize
from inverted_index import *

""" Pre-process the query the same way documents are processed
"""
def pre_process_query(query):
    new_query = word_tokenize(query)
    new_query = [w for w in new_query if w.isalnum()]
    new_query = [x.lower() for x in new_query]
    return new_query

""" Implementation of query retrieval system that uses lnc-ltc scoring scheme

Reads all the data related to inverted index and necessary for computation
"""
def read_data_structures():
    global titles
    global idnos
    global document_frequencies
    global document_bodies
    global document_token_list
    global inverted_index
    global document_weights

    with open( 'index_data/titles.data', 'rb') as f:
        titles = pickle.load(f)
    with open( 'index_data/idnos.data', 'rb') as f:
        idnos = pickle.load(f)
    with open( 'index_data/inverted_index.data', 'rb') as f:
        inverted_index = pickle.load(f)
    with open( 'index_data/document_bodies.data', 'rb') as f:
        document_bodies = pickle.load(f)
    with open( 'index_data/document_frequencies.data', 'rb') as f:
        document_frequencies = pickle.load(f)
    with open( 'index_data/document_token_list.data', 'rb') as f:
        document_token_list = pickle.load(f)
    with open( 'index_data/document_weights.data', 'rb') as f:
        document_weights = pickle.load(f)

""" Return all the query terms with the corresponding count
"""
def get_query_terms(query):
    return Counter(query)

""" Generate normalized query scores

The scoring scheme is ltc
l -> Logarithmic tf
t -> idf
c -> Cosine normalization
"""
def get_normalized_query_scores(query_terms):
    # Calculate logarithmic tf
    tf_idf = {}
    N = len(document_frequencies)
    for term in query_terms:
        tf = 1 + math.log10(query_terms[term]) # Assign TF
        idf = 0
        if term in inverted_index.keys():
            idf = math.log10(N / len(inverted_index[term]))
        else:
            idf = 0
        tf_idf[term] = tf*idf # Find TF-IDF for each term
    cosine = math.sqrt(sum([x**2 for x in tf_idf.values()]))
    if cosine!=0:
        cosine = 1 / cosine
    for term in tf_idf:
        tf_idf[term] = tf_idf[term] * cosine    # cosine normalize the tf_idf scores for each term in the query
    return tf_idf               # returns a dict of terms vs tf_idf scores

""" Function that returns the weight of a given term in document

Returns weight of term if it is in the dictionary. Else, it returns 0.
"""
def get_document_term_weight(term, term_weights):
    if term in term_weights.keys():
        return term_weights[term]
    else:
        return 0

""" Computes the cosine similarity between query and document weights

Returns a sorted list of documents with their scores in non-increasing order
"""
def compute_scores(query_wt, document_wt):
    scores = [[i, 0] for i in range(len(document_wt))]  #list of [index, score] pairs
    for i in range(len(document_wt)):
        doc_tf = document_wt[i]                         #list of term - frequency pairs for one document
        score = 0
        for term in query_wt.keys():
            score += query_wt[term]*get_document_term_weight(term,doc_tf)
        scores[i] = [i, score]                          #a pair with of an index vs score
    return scores

""" Performs search

Accepts user query, and ranks documents based on cosine similarity, then prints
the top 10 results based on the rank
"""
def search():
    query = input('Enter your query: ')
    # Pre-process the query
    processed_query = pre_process_query(query)
    query_terms = get_query_terms(processed_query)
    print("Query Terms: ", query_terms)

    # Find query and document weights
    query_wt = get_normalized_query_scores(query_terms)

    # Find the scores of each document
    scores = compute_scores(query_wt, document_weights)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Checking if number of results is less than 10
    num = len(titles) if len(titles)<10 else 10

    print("The top ", num, " documents matching with the query '", query, "' are:")
    for i in range(10):
        if i == len(titles):
            break
        print(str(i+1) + ". Document " + str(idnos[scores[i][0]]) + ": " + str(titles[scores[i][0]]) + ", Score: " + str(round(scores[i][1], 3)))
