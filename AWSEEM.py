from preprocessing import load_document_file, preprocess, preprocess_query
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
import pandas as pd
import scipy.sparse as sparse
import csv


def search_evaluation(query: str, search_engine, t=20, b=200, s=50, synonyms=7):
    """
    :param query: The given query to collect documents with
    :param search_engine: The search engine to be used
    :param t: The top t documents returned by the document
    :param b: The top b documents used to get top s similar documents. b documents are collected per query/synonym
    :param s: The most similar s documents assumed to be relevant by AWSEEM. The paper tested 50 and 100.
    :param synonyms: The amount of synonyms which should be used for collecting documents for query similarity
    :return: Tuple (p, r) with the effectiveness measured with precision and recall
    """
    # Preprocessing is assumed to have already been done

    # TODO Collect top b results from the query using the search engine
    print(query)
    pp_query = preprocess_query(query)
    print(pp_query)

    filename = "documents.json"
    documents = load_document_file(filename)
    query_results = {k: v for (k, v) in documents.items() if "håndbold" in v}

    preprocess(load_filename=query_results, word_save_filename="Generated Files/words.csv", doc_save_filename="Generated Files/docs.csv", doc_word_save_filename="Generated Files/docword.csv", doc_word_matrix_save_filename="Generated Files/cvmatrix.npz")
    cv_matrix = sparse.load_npz("Generated Files/cvmatrix.npz")

    # TODO Retrieve top t results from the top b
    #D_t = [query_results[i] for i in [1, 2, 3, 7, 8, 27, 31, 34, 72, 73]]

    # TODO Retrieve synonyms for the query

    # TODO Collect top b results from queries of synonyms using the search engine

    # TODO Merge the collected results

    # TODO Sort the merged results using their similarity to the original query (Vector product formula, TF-IDF)
    # TF-IDF
    #tfidfvectorizer = TfidfVectorizer()
    #X = tfidfvectorizer.fit_transform(query_results)
    tfidftransformer = TfidfTransformer()
    X = tfidftransformer.fit_transform(cv_matrix)

    with open("Generated Files/words.csv", "r") as file:
        reader = csv.reader(file)
        index_word_list = list(reader)
    word_list = [w for (i, w) in index_word_list]

    # Table for visual TF-IDF values
    df = pd.DataFrame(X[0].T.todense(), index=word_list, columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    print(df.head(25))

    word_tfidf = dict(zip(word_list, X[0].T.todense()))
    query_tfidf = {k: v for (k, v) in word_tfidf if k in pp_query}
    # TODO Retrieve top s documents from the results

    # TODO Check if top t documents (D_t) appear in top s similar documents (D_s)

    # TODO Calculate precision and recall
    pass


if __name__ == '__main__':
    query = "er der håndbold kvinder kamp"
    search_engine = "insert pagerank"
    search_evaluation(query, search_engine)
