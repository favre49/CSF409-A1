import numpy
import nltk # regex also needs to be installed
from bs4 import BeautifulSoup
import pickle
from collections import Counter
import os

nltk.download('punkt')

ROOT_DIR = "./data/Wikipedia/AO"

"""Retrieves documents from a given file

Uses BeautifulSoup to parse the file and extract the titles, ids, and document bodies
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

TODO: consider removing the href links in the extraction part,
probably faster? But more coupled
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

def create_inverted_index(document_frequencies, idnos):
    inverted_index ={}
    assert len(document_frequencies) == len(idnos), "Lengths don't match!"
    for document_frequency, idno in zip(document_frequencies,idnos):
        for key in inverted_index.keys():
            if key in inverted_index.keys():
                inverted_index[key].append((idno,document_frequency[key]))
            else:
                inverted_index[key] = []
                inverted_index[key].append((idno,document_frequency[key]))
    return inverted_index

# Testing
if __name__ == '__main__':
    titles, idnos, document_bodies = [],[],[]
    count = 0
    for subdir, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            count = count+1
            print(count)
            file_titles,file_idnos,file_document_bodies = get_document_from_file(os.path.join(subdir,file))
            titles = titles + file_titles
            idnos = idnos + file_idnos
            document_bodies = document_bodies + file_document_bodies
    document_token_list = []
    for document in document_bodies:
        tokens = preprocess_text(document)
        document_token_list.append(tokens)
    document_frequencies = [Counter(token_list) for token_list in document_token_list]
    inverted_index = create_inverted_index(document_frequencies, idnos)
    with open('data/document_bodies.data','wb') as f:
        pickle.dump(document_bodies,f)
    with open('data/titles.data','wb') as f:
        pickle.dump(titles,f)
    with open('data/document_token_list.data','wb') as f:
        pickle.dump(document_token_list,f)
    with open('data/inverted_index.data','wb') as f:
        pickle.dump(inverted_index,f)
