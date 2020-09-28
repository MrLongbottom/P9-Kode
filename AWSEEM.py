from preprocessing import load_document_file
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


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
    filename = "documents.json"
    documents = load_document_file(filename)
    query_results = [v for (k, v) in documents.items() if "håndbold" in v]

    # TODO Retrieve top t results from the top b
    D_t = [query_results[i] for i in [1, 2, 3, 7, 8, 27, 31, 34, 72, 73]]

    # TODO Retrieve synonyms for the query

    # TODO Collect top b results from queries of synonyms using the search engine

    # TODO Merge the collected results

    # TODO Sort the merged results using their similarity to the original query (Vector product formula, TF-IDF)
    # TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(query_results)
    print(vectorizer.get_feature_names())
    print(X.shape)
    df = pd.DataFrame(X[0].T.todense(), index=vectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    print(df.head(25))

    # TODO Retrieve top s documents from the results

    # TODO Check if top t documents (D_t) appear in top s similar documents (D_s)

    # TODO Calculate precision and recall
    pass


if __name__ == '__main__':
    query = "håndbold kvinder kamp"
    search_engine = "insert pagerank"
    search_evaluation(query, search_engine)
