from typing import Dict

import numpy as np
import pandas
import scipy.sparse as sp
from gensim.corpora import Dictionary

from LDA.lda import load_lda, get_document_topics_from_model, load_corpus
from preprocessing import generate_queries
from standard_random_walk import random_walk_with_teleport


def make_personalization_vector(topics: Dict[int, float], topic_doc_matrix):
    vector = np.zeros(topic_doc_matrix.shape[0])
    for key, value in topics.items():
        vector[topic_doc_matrix.getrow(key).nonzero()[1]] = value
    return vector


def query_topics(query: str, model_path: str, topic_doc_path) -> np.ndarray:
    lda = load_lda(model_path)
    topic_doc_matrix = sp.load_npz(topic_doc_path)
    corpus = Dictionary(load_corpus("Generated Files/corpus"))
    q_topics = get_document_topics_from_model(query, lda, corpus)
    return make_personalization_vector(q_topics, topic_doc_matrix)


def search(size_of_adj: int, lda_path: str, vectorizer_path,  topic_doc_matrix_path: str, adj_matrix_path):
    adj_matrix = sp.load_npz(adj_matrix_path)[:size_of_adj, :size_of_adj]
    df = pandas.read_csv("Generated Files/word2vec.csv", header=None)
    words = dict(zip(df[0], df[1]))
    queries = generate_queries(sp.load_npz(vectorizer_path)[:500, :500], words, 10, 4)

    for query in queries.items():
        search_vector = query_topics(query[1].split(' '), lda_path, topic_doc_matrix_path)
        doc_ranks = random_walk_with_teleport(adj_matrix, search_vector[:size_of_adj])
        print(f" Query: {query[1]} Hit: {list(doc_ranks).index(query[0])} ")


if __name__ == '__main__':
    search(500, "LDA/model/2017_model",
           "Generated Files/count_vec_matrix.npz",
           "Generated Files/topic_doc_matrix.npz",
           "new_matrix.npz")