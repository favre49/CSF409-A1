import numpy
import nltk # regex also needs to be installed
from bs4 import BeautifulSoup
import pickle
from collections import Counter
import os
import math

nltk.download('punkt')

ROOT_DIR = "./data/Wikipedia/AO" # Root directory for input files
INDEX_DATA_DIR = "./index_data" # Directory for serialized data

"""Retrieves documents from a given file

Extracts the titles, ids and bodies of every document in a file
"""
def get_document_from_file(filename):
    with open(filename,'r') as input_file:
        soup = BeautifulSoup(input_file,'html.parser')  # Use BeautifulSoup to parse the file
        documents = soup.findAll("doc")
        titles = [document["title"] for document in documents]  # Get titles
        ids = [document["id"] for document in documents]  # Get id numbers
        document_bodies = [BeautifulSoup(str(document),'html.parser').get_text() for document in documents]  # Get the document bodies
    return titles, ids, document_bodies

"""Preprocess the text

This function takes the document body as an input. It performs the following
preprocessing steps and returns a list of tokens:
    1. Removes all hyperlinks
    2. Removes punctuation
    3. Converts all tokens to lowercase
"""
def preprocess_text(text):
    soup = BeautifulSoup(text,'html.parser')
    # Remove all href links
    for a in soup.findAll('a'):
        a.replaceWithChildren()
    text = soup.get_text()
    tokens = nltk.word_tokenize(text)  # Tokenize text
    tokens = [token for token in tokens if token.isalnum()]  # Remove punctuation
    tokens = [token.lower() for token in tokens]  # Convert all tokens to lowercase
    return tokens

"""Creates the inverted index

The inverted index is implemented as a Python `dict`, in which the key is a token
and the value is a list of (idno,frequency) tuples
"""
def create_inverted_index(document_frequencies, idnos):
    inverted_index ={}
    for document_frequency, idno in zip(document_frequencies,idnos):
        for key in document_frequency.keys():
            if key in inverted_index.keys():
                inverted_index[key].append((idno,document_frequency[key]))
            else:
                inverted_index[key] = []
                inverted_index[key].append((idno,document_frequency[key]))
    return inverted_index

""" Generate normalized document scores

Scoring scheme: lnc
l -> Logarithmic tf
n -> No idf
c -> Cosine normalization
"""
def get_normalized_doc_weights(document_frequencies):
    doc_weights = [{} for i in range(len(document_frequencies))] #list of as many dicts as the number of documents
    # Finding logarithmic tf
    for i in range(len(document_frequencies)):
        for term in document_frequencies[i].keys():
            val = document_frequencies[i][term]
            doc_weights[i][term] = 1 + math.log10(val)
    # Applying cosine normalization
    for i in range(len(doc_weights)):
        doc_tf = doc_weights[i]
        square_sum = math.sqrt(sum([v ** 2 for v in doc_tf.values()]))
        if square_sum != 0:
            factor = 1 / square_sum
        else:
            factor = 0
        for term in doc_weights[i].keys():
            doc_weights[i][term] *= factor
    return doc_weights

""" Serializes the inverted index and associated data

The associated data serialized is:
    1. Document titles
    2. Document ID numbers
    3. Document bodies
    4. Document token frequencies
    5. Document token list
    6. Document term weights
"""
def generate_inverted_index() :
    print("Generating Inverted Index...")
    titles, idnos, document_bodies = [],[],[]
    cnt = 0;
    # Iterate over all files and extract document
    for subdir, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            cnt += 1
            print(cnt)
            file_titles,file_idnos,file_document_bodies = get_document_from_file(os.path.join(subdir,file))
            titles = titles + file_titles
            idnos = idnos + file_idnos
            document_bodies = document_bodies + file_document_bodies

    # Tokenize all the documents
    document_token_list = []
    for document in document_bodies:
        tokens = preprocess_text(document)
        document_token_list.append(tokens)

    # Find document frequencies
    document_frequencies = [Counter(token_list) for token_list in document_token_list]

    # Create inverted index
    inverted_index = create_inverted_index(document_frequencies, idnos)
    
    # Precompute document weights for later use
    document_weights = get_normalized_doc_weights(document_frequencies)

    # Store since it takes a long time to build
    with open('index_data/titles.data','wb') as f:
        pickle.dump(titles,f)
    with open('index_data/idnos.data','wb') as f:
        pickle.dump(idnos,f)
    with open('index_data/document_token_list.data','wb') as f:
        pickle.dump(document_token_list,f)
    with open('index_data/document_frequencies.data','wb') as f:
        pickle.dump(document_frequencies,f)
    with open('index_data/document_bodies.data','wb') as f:
        pickle.dump(document_bodies,f)
    with open('index_data/inverted_index.data','wb') as f:
        pickle.dump(inverted_index,f)
    with open('index_data/document_weights.data','wb') as f:
        pickle.dump(document_weights,f)

    print("Generated Inverted Index!")
