from search import *
from search_modified import *
import search

""" Reads serialized data for searching
"""
def get_data_structures_OkapiBM25():
    global titles
    global idnos
    global document_frequencies
    global document_bodies
    global document_token_list
    global inverted_index
    global document_weights

    titles = search.titles
    idnos = search.idnos
    document_frequencies = search.document_frequencies
    document_bodies = search.document_bodies
    document_token_list = search.document_token_list
    inverted_index = search.inverted_index
    document_weights = search.document_weights

""" Performs a query with the OkapiBM25+ algorithm

The hyperparameters of the equation k,b and delta re passed as function
arguments.
"""
def query_OkapiBM25(query_terms,k,b,delta):
    N = len(document_bodies)
    avg_length = sum([len(doc) for doc in document_token_list])
    avg_length /= N
    doc_scores = [[i,0] for i in range(N)]
    for i in range(0,len(document_frequencies)):
        freq_dict = dict(document_frequencies[i])
        score = 0
        for term in query_terms:
            if term in freq_dict:
                df = len(inverted_index[term])
                tf = freq_dict[term]
                term_score = (k+1)*tf
                document_length = len(document_token_list[i])
                term_score /= k*((1-b)+b*document_length/avg_length)+tf
                term_score += delta
                term_score *= math.log10(N+1/df)
                score += term_score
        doc_scores[i] = [i,score]
    doc_scores = sorted(doc_scores, key = lambda x : x[1], reverse=True) 
    return doc_scores

""" Performs a search with the Okapi BM25+ algorithm
"""
def search_OkapiBM25():
    query = input('Please enter your query: ')
    processed_query = pre_process_query(query)
    spell_corrected_query = spelling_corrector(processed_query) # Spell correct query
    query_terms = get_query_terms(spell_corrected_query)
    print("Query Terms: ", query_terms)

    # Okapi BM25+ hyperparameters
    k = 0.5
    b = 0.5
    delta = 1
    scores = query_OkapiBM25(query_terms,k,b,delta)
    
    # Checking if number of results is less than 10
    num = len(titles) if len(titles)<10 else 10

    print("The top ", num, " documents matching with the query '", query, "' are:")
    for i in range(10):
        if i == len(titles):
            break
        print(str(i) + ". Document " + str(idnos[scores[i][0]]) + ": " + str(titles[scores[i][0]]) + ", Score: " + str(round(scores[i][1], 3)))
