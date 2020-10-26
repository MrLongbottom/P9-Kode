from typing import Dict

import numpy as np
import scipy.sparse as sp
from gensim.corpora import Dictionary

from LDA.lda import load_lda, get_document_topics_from_model, load_corpus
from cluster_random_walk import cluster_page_rank
from preprocessing import preprocess_query


def make_personalization_vector(topics: Dict[int, float], topic_doc_matrix):
    vector = np.zeros(topic_doc_matrix.shape[0])
    for key, value in topics.items():
        vector[topic_doc_matrix.getrow(key).nonzero()[1]] = value
    return vector


def query_topics(query: str, model_path: str, topic_doc_path) -> np.ndarray:
    processed_query = preprocess_query(query)
    lda = load_lda(model_path)
    topic_doc_matrix = sp.load_npz(topic_doc_path)
    corpus = Dictionary(load_corpus("Generated Files/corpus"))
    q_topics = get_document_topics_from_model(processed_query, lda, corpus)
    return make_personalization_vector(q_topics, topic_doc_matrix)


def search(query: str, size_of_adj: int, lda_path: str, topic_doc_matrix_path: str) -> np.ndarray:
    search_vector = query_topics(query, lda_path, topic_doc_matrix_path)
    adj_matrix = sp.load_npz("Generated Files/full_matrix.npz")[:size_of_adj, :size_of_adj]
    return cluster_page_rank(adj_matrix, search_vector[:size_of_adj])


if __name__ == '__main__':
    query1 = "fodbold spiller"
    query2 = "katte i vand"

    r_list1 = search(query1, 100, "LDA/model/docu_model_sqrt_div2", "Generated Files/topic_doc_matrix.npz")
    r_list2 = search(query2, 100, "LDA/model/docu_model_sqrt_div2", "Generated Files/topic_doc_matrix.npz")

    print(r_list1[:10])
    print(r_list2[:10])
