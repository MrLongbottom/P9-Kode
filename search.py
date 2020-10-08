from typing import Dict

import numpy as np
import scipy.sparse as sp

from LDA.lda import load_lda, get_document_topics_from_model
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
    q_topics = get_document_topics_from_model(processed_query, lda)
    return make_personalization_vector(q_topics, topic_doc_matrix)


if __name__ == '__main__':
    query = "fodbold spiller"
    query2 = "katte i vand"

    search_vector = query_topics(query2, "LDA/model/search_model", "Generated Files/search/topic_doc_matrix.npz")
    adj_matrix = sp.load_npz("Generated Files/full_matrix.npz")[:1000, :1000]
    ranked_list = cluster_page_rank(adj_matrix, search_vector[:1000])[:10]
    print(ranked_list)
