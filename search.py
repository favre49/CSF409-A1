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
    tf = {}
    idf = {}
    tf_idf = {}
    N = len(document_frequencies)
    for term in query_terms:
        tf[term] = 1 + math.log10(query_terms[term])
    for term in query_terms:
        if term in inverted_index.keys():
            idf[term] = math.log10(N / len(inverted_index[term]))
        else:
            idf[term] = 0
    for term in query_terms:    #find tf_idf scores for each term in the query
        tf_idf[term] = tf[term]*idf[term]
    cosine = math.sqrt(sum([x**2 for x in tf_idf.values()]))
    if cosine!=0:
        cosine = 1 / cosine
    for term in tf_idf:
        tf_idf[term] = tf_idf[term] * cosine    #cosine- normalize the tf_idf scores for each term in the query
    return tf_idf               #returns a dict of terms vs tf_idf scores

""" Generate normalized document scores

Scoring scheme: lnc
l -> Logarithmic tf
n -> No idf
c -> Cosine normalization
"""
def get_normalized_doc_weights(query):
    doc_weights = [[] for i in range(len(document_frequencies))] #list of as many lists as the number of documents
    # Finding logarithmic tf
    for i in range(len(document_frequencies)):
        for term in document_frequencies[i].keys():
            val = document_frequencies[i][term]
            doc_weights[i].append([term, 1 + math.log10(val)])
    # Applying cosine normalization
    normalized_doc_weights = [[] for i in range(len(doc_weights))]
    for i in range(len(doc_weights)):
        doc_tf = doc_weights[i]
        square_sum = math.sqrt(sum([v[1] ** 2 for v in doc_tf]))
        if square_sum != 0:
            factor = 1 / square_sum
        else:
            factor = 0
        for j in range(len(doc_tf)):
            normalized_doc_weights[i].append([doc_tf[j][0], doc_tf[j][1] * factor])
    return normalized_doc_weights   #returns a list of [doc_terms, number of occurrences of the term in document]

""" Function that returns the weight of a given term in query

Returns 0 if term not in query
"""
def get_query_term_weight(term, term_weights):
    if term in term_weights.keys():
        return term_weights[term]
    else:
        return 0

""" Computes the cosine similarity between query and document weights

Returns a sorted list of documents with their scores in non-increasing order
"""
def compute_scores(query_wt, document_wt):
    scores = [[i, 0] for i in range(len(document_wt))]  #list of [rank, score] pairs
    for i in range(len(document_wt)):
        doc_tf = document_wt[i]                         #list of term - frequency pairs for one document
        score = 0
        for j in range(len(doc_tf)):                    #looping through the total number of terms of the document
            term = doc_tf[j][0]
            term_weight = get_query_term_weight(term, query_wt) #gets query term weight corresponding to each term in the document.
            score += term_weight * doc_tf[j][1]         #adds product of query and doc weights of the term to the score
        scores[i] = [i, score]                          #a pair with of an index vs score
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
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

    # Find query and docuemnt weights
    query_wt = get_normalized_query_scores(query_terms)
    document_wt = get_normalized_doc_weights(query_terms)

    # Find the ranking
    scores = compute_scores(query_wt, document_wt)

    # Checking if number of results is less than 10
    num = len(titles) if len(titles)<10 else 10

    print("The top ", num, " documents matching with the query '", query, "' are:")
    for i in range(10):
        if i == len(titles):
            break
        print(str(i) + ". Document " + str(idnos[scores[i][0]]) + ": " + str(titles[scores[i][0]]) + ", Score: " + str(round(scores[i][1], 3)))
