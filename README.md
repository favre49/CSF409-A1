# CSF409-A1
Assignment for CSF409 Information Retrieval

## Group Members

|Name|ID|
|---|---|
|Rahul Ganesh Prabhu|2018A7PS0193P|
|Harsh Khatri |2017A7PS0055P|
|Rishav Das |2018A7PS0157P|
|B. Rishishankar |2018A4PS0549P|
|Yaganti Sivakrishna|2017A7PS0045P|

## Instructions to run

Download the provided data and place it in `data/`. The datapath `data/Wikipedia/AO/` should be defined as it is the corpus being used.

Run `pip install -r requirements.txt` to install the dependencies.

To test on queries, run `python test_queries.py`

## Code description

The inverted index and any other necessary data is created and saved in `inverted_index.py`. 

Part 1 is written in `search.py`. It uses a lnc-ltc scoring scheme to rank the most relevant documents

Part 2 has two sub parts:

1. This can be found in `search_modified.py`. It find the synonym sets of each query term and calculates the score for all the synonyms. A weight is assigned to the score generated from the synonym sets.
2. This can be found in `okapi_bm25.py`. It uses Okapi BM25+ to find the most relevant documents to the query. It uses the spelling correction implemented in subpart 1 as well.

This code can be found [on GitHub](https://github.com/favre49/CSF409-A1). It will be made public 48 hours after submission.
