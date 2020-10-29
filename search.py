from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import sklearn
from sklearn.preprocessing import normalize
from fast_pagerank import pagerank
import numpy as np
import pandas
import scipy.sparse as sp
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from scipy.spatial import distance
from tqdm import tqdm

import preprocessing
import standard_random_walk
from LDA.lda import load_lda, get_document_topics_from_model, load_corpus
from preprocessing import generate_queries
from standard_random_walk import random_walk_with_teleport


def make_personalization_vector(word: str, topic_doc_matrix, corpus: Dictionary, lda: LdaModel):
    """
    Making a personalization vector based on the topics.
    Gives each documents in the document a score based on the distribution value
    :param lda: the lda model
    :param corpus: a dictionary over the whole document set
    :param word: a word
    :param topic_doc_matrix: topic document matrix
    :return: np.ndarray
    """
    topics = get_document_topics_from_model([word], lda, corpus)
    vector = np.zeros(topic_doc_matrix.shape[0])
    for key, value in topics.items():
        vector[topic_doc_matrix.getrow(key).nonzero()[1]] = value
    return vector


def make_personalization_vector_word_based(word: str, dt_matrix, tw_matrix):
    """
    Takes a query and transforms it into a vector based on each words topic distribution
    For each word we find its topic distribution and compare it against every other document
    using jensen shannon distance.
    :param word: a word
    :param dt_matrix: topic document matrix
    :param lda: the lda model
    :return: a personalization vector (np.ndarray)
    """
    # Initialization
    p_vector = np.zeros(dt_matrix.shape[0])
    vector = np.zeros(dt_matrix.shape[1])

    # Getting the word index and then getting the topic distribution for that given word
    word_index = [key for key, value in word2vec.items() if value == word][0]
    word_vector = tw_matrix.T[word_index]
    word_vector = sklearn.preprocessing.normalize(word_vector, norm='l1', axis=1)
    return word_vector


def query_topics(query: List[str], lda_model, dt_matrix, tw_matrix, word2vec) -> np.ndarray:
    """
    Takes a list of words and makes a personalization vector based on these
    :param query: list of words
    :param model_path: lda model path
    :param dt_matrix: topic document matrix path
    :param indexes: used for comparison function
    :return: a personalization vector
    """
    # combine topic distributions for each word in query
    p_vector = np.zeros(83)
    for word in query:
        if word in word2vec.values():
            p_vector += make_personalization_vector_word_based(word, dt_matrix, tw_matrix)
        else:
            # Todo needs to be cut
            p_vector += make_personalization_vector(word, dt_matrix, tw_matrix, word2vec, lda_model)
    p_vector = p_vector / len(query)
    doc_sim = np.zeros(dt_matrix.shape[0])
    for index, doc in enumerate(dt_matrix):
        doc_sim[index] = 1 - distance.jensenshannon(np.array(p_vector)[0], doc.toarray()[0])
    return sklearn.preprocessing.minmax_scale(doc_sim)


def search(lda_model, count_vec, adj_matrix, dt_matrix, tw_matrix, word2vec):
    """
    Our search algorithm.
    Takes a size if you want to search a smaller part of the matrix.
    It generates 10 queries randomly from the document set.
    And evaluates the given article by printing its index in the ordered list.
    :param size_of_adj: number of documents you want to search
    :param lda_path: lda model path
    :param vectorizer_path: the given vectorize
    :param topic_doc_matrix_path: topic document path
    :param adj_matrix_path: adjacency matrix path
    :return: prints the index of the articles in the search algorithm
    """
    queries = generate_queries(count_vec, word2vec, 10, 4)
    for doc, query in queries.items():
        p_vector = query_topics(query.split(' '), lda_model, dt_matrix, tw_matrix, word2vec)
        doc_ranks = pagerank(sp.csr_matrix(adj_matrix), personalize=p_vector[:size_of_adj])
        print(f" Query: {query}")
        print(f"PageRank Hit: {list(doc_ranks.argsort()[:][::-1]).index(doc)}")
        print(f"P_vector Hit: {list(p_vector.argsort()[:][::-1]).index(doc)}")


if __name__ == '__main__':
    adj_matrix = sp.load_npz("Generated Files/adj_matrix.npz")
    size_of_adj = min(adj_matrix.shape[0], adj_matrix.shape[1])
    adj_matrix = adj_matrix[:size_of_adj, :size_of_adj]
    adj_matrix = standard_random_walk.construct_transition_probability_matrix(adj_matrix.toarray())
    dt_matrix = sp.load_npz("Generated Files/topic_doc_matrix.npz")
    tw_matrix = sp.load_npz("Generated Files/topic_word_matrix.npz")
    # TODO make query generation in separate file
    count_vec = sp.load_npz("Generated Files/count_vec_matrix.npz")
    count_vec = count_vec[:2000]
    lda_model = load_lda("LDA/model/document_model")
    word2vec = preprocessing.load_vector_file("Generated Files/word2vec.csv")

    search(lda_model, count_vec, adj_matrix, dt_matrix, tw_matrix, word2vec)
