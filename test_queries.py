from search import search, read_data_structures
from inverted_index import generate_inverted_index
from okapi_bm25 import search_OkapiBM25, get_data_structures_OkapiBM25
from search_modified import modified_search, get_data_structures

if __name__ == '__main__':
    # generate_inverted_index()
    read_data_structures()
    get_data_structures()
    get_data_structures_OkapiBM25()

    while(True):
        print("Choose which search method to employ")
        print("1. lnc-ltc scoring scheme from Part-1")
        print("2. Modified search scheme from Part-2, using synonyms from entered queries")
        print("3. To search using the Okapi BM25+ model from Part-2")
        print("0. Exit")

        choice = int(input("Enter your request: "))
        print("\n")
        if choice == 1:
            search()
        elif choice == 2:
            modified_search()
        elif choice == 3:
            search_OkapiBM25()
        else:
            break

