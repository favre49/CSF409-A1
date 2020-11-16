from search import *
import search
import nltk
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
nltk.download('wordnet')

# This file improves the previously implemented information retrieval system which was based on the lnc-ltc scoring scheme
# Modifications have been made in searching and preprocessing of the queries
# A Spelling correction heuristic has been added during the preprocessing of queries
# For further improvement in results, the modified retrieval system also searches for words having similar meaning to the query term

def get_data_structures():
    global titles
    global idnos
    global document_frequencies
    global document_bodies
    global document_token_list
    global inverted_index

    titles = search.titles
    idnos = search.idnos
    document_frequencies = search.document_frequencies 
    document_bodies = search.document_bodies
    document_token_list = search.document_token_list
    inverted_index = search.inverted_index


# This function implements a heuristic for checking the spelling of query terms.
# Every query term is spell checked and if a term has been misspelt then the user is asked when the user wants to search for the corrected query term instead

def spelling_corrector(query):
    spelling = SpellChecker()
    misspelt = spelling.unknown(query)
    print("Executing Spelling Check ...")
    if misspelt:
        for w in query:
            if w in misspelt:
                print("Did you mean " + spelling.correction(w) + " instead of " + w + " ? Press y for Yes: ")
                choice = input()
                if choice == 'y':
                    query[:] = [spelling.correction(w) if x == w else x for x in query]
    return query


# This function finds the relevant synonyms of the terms in the query

def compute_synonyms(query_tokens):
    synonym_set = []
    for word in query_tokens:
        s = [word]
        for l in wordnet.synsets(word):
            for w in l.lemma_names():
                s.append(w)
        synonym_set.append(list(set(s)))
    return synonym_set


# This function computes the cosine similarity between query and document weights
# An unsorted list of the scores of the scores is being returned this time

def compute_scores(query_wt, document_wt):
    scores = [[i, 0] for i in range(len(document_wt))]              #list of 2-element lists [rank, score]
    for i in range(len(document_wt)):
        doc_tf = document_wt[i]                                     #list of term-frequency pairs for one document
        score = 0
        for j in range(len(doc_tf)):                                #looping through the total number of terms of the document
            term = doc_tf[j][0]
            term_weight = get_query_term_weight(term, query_wt)     #getting the query term weight corresponding to each term in the document.
            score += term_weight * doc_tf[j][1]                     #adding product of query and doc weights of the term to the score
        scores[i] = [i, score]                                      #a pair with of an index vs score
    
    return scores


# This function merges existing scores with the newly computed scores of the synonym set of a term

def compute_merged_scores(cscores, wt1, wt2):

    # Here w1 = weight associated with score of original query terms and w2 = weight associated with score of all the synonyms of original term

    original_scores = cscores[0]
    new_scores = [ [i, original_scores[i][1]*wt1] for i in range(len(original_scores))]

    if len(cscores) == 1:
        return new_scores
    else:
        for i in range(len(new_scores)):
            for j in range(1, len(cscores)):
                new_scores[i][1] += (wt2 * (cscores[j])[i][1])
        return new_scores


# This function implements the modified search in the improved info retrieval system
# User query is accepted and documents are ranked on the basis of cosine similarity after which the top 10 results are printed

def modified_search():

    query = input('Enter query: ')

    # Pre-processing the query
    processed_query = pre_process_query(query)

    # Doing spell correction 
    spell_corrected_query = spelling_corrector(processed_query)

    query_terms = get_query_terms(spell_corrected_query)
    print("Final Query Terms: ", query_terms)

    # Computing the query and document weights
    query_wt = get_normalized_query_scores(query_terms)
    document_wt = get_normalized_doc_weights(query_terms)

    scores = []

    # Computing Cosine Scores. An unsorted list is returned here
    original_scores = compute_scores(query_wt, document_wt)
    scores.append(original_scores)

    # Generating a synonym set for each query term
    query_syn_set = compute_synonyms(query_terms)

    for qs in query_syn_set:
        new_query_terms = Counter(qs)
        new_q_wt = get_normalized_query_scores(new_query_terms)
        new_d_wt = get_normalized_doc_weights(new_query_terms)

        # Note: The weight of the scores of the synonym sets can be changed. Currently, the weight has been set to 0.2 after repeated evaluation of performance.

        new_score = compute_scores(new_q_wt, new_d_wt)
        scores.append(new_score)

    # Here modified score = original_score + weight*new_score
    modified_score = compute_merged_scores(scores, wt1=1, wt2=0.2)

    # Sorting the modified scores to get the final scores
    final_scores = sorted(modified_score, key=lambda x: x[1], reverse=True)

    # Checking if number of results is not less than 10
    num = len(titles) if len(titles)<10 else 10

    print("The top ", num, " documents matching with the query '", query, "' are:")
    for i in range(10):
        if i == len(titles):
            break
        print(str(i) + ". Document " + str(idnos[final_scores[i][0]]) + ": " + str(titles[final_scores[i][0]]) + ", Score: " + str(round(final_scores[i][1], 3)))
        print(document_bodies[final_scores[i][0]])
