from search import *

def query_OkapiBM25(query_terms,k,b):
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
                term_score = math.log10(N/df)*(k+1)*tf
                document_length = len(document_token_list[i])
                term_score /= k*((1-b)+b*document_length/avg_length)+tf
                score += term_score
        doc_scores[i] = [i,score]
    doc_scores = sorted(doc_scores, key = lambda x : x[1], reverse=True) 
    return doc_scores


def search_Okapi_BM25():
    query = input('Please enter your query: ')
    processed_query = pre_process_query(query)
    query_terms = get_query_terms(query_processed)
    print("Query Terms: ", query_terms)

    # Okapi BM25 hyperparameters
    k = 0.5
    b = 0.5
    scores = query_OkapiBM25(query_terms,k,b)
    
    # Checking if number of results is less than 10
    num = len(titles) if len(titles)<10 else 10

    print("The top ", num, " documents matching with the query '", query, "' are:")
    for i in range(10):
        if i == len(titles):
            break
        print(str(i) + ". Document " + str(idnos[scores[i][0]]) + ": " + str(titles[scores[i][0]]) + ", Score: " + str(
            round(scores[i][1], 3)))
        print(document_bodies[scores[i][0]])
