from typing import Dict, List

import numpy as np
import pandas
import scipy.sparse as sp
from gensim.corpora import Dictionary
from scipy.spatial import distance
from tqdm import tqdm

from LDA.lda import load_lda, get_document_topics_from_model, load_corpus
from preprocessing import generate_queries
from standard_random_walk import random_walk_with_teleport


def make_personalization_vector(topics: Dict[int, float], topic_doc_matrix):
    """
    Making a personalization vector based on the topics.
    Gives each documents in the document a score based on the distribution value
    :param topics: the topics and the distributions
    :param topic_doc_matrix: topic document matrix
    :return: np.ndarray
    """
    vector = np.zeros(topic_doc_matrix.shape[0])
    for key, value in topics.items():
        vector[topic_doc_matrix.getrow(key).nonzero()[1]] = value
    return vector


def make_personalization_vector_word_based(query: List[str], topic_doc_matrix, lda, index):
    """
    Takes a query and transforms it into a vector based on each words topic distribution
    For each word we find its topic distribution and compare it against every other document
    using jensen shannon distance.
    :param query: the list of strings
    :param topic_doc_matrix: topic document matrix
    :param lda: the lda model
    :param index: an index used to do comparisons, (normally not used)
    :return: a personalization vector (np.ndarray)
    """
    p_vector = np.zeros(topic_doc_matrix.shape[0])
    vector = np.zeros(topic_doc_matrix.shape[1])
    df = pandas.read_csv("Generated Files/word2vec.csv", header=None)
    words = dict(zip(df[0], df[1]))
    topic_word_matrix = lda.get_topics()
    original = []
    for word in query:
        word_index = [key for key, value in words.items() if value == word][0]
        # original.append(topic_word_matrix.T[word_index])
        vector += topic_word_matrix.T[word_index]
    for index, doc in tqdm(enumerate(topic_doc_matrix)):
        p_vector[index] = 1 - distance.jensenshannon(vector, doc.toarray()[0])
    # compare = compare_vectors(topic_doc_matrix, index, vector, original)
    return p_vector


def compare_vectors(topic_doc_matrix, index, vector, original):
    """
    Stacks 4 vectors on top of each other for comparison.
    1: the document topic distribution
    2: the personalization vector for a given query
    3-4: the topic distribution for each word in the query
    (only works for queries of length 2)
    :param topic_doc_matrix: topic document matrix
    :param index: index of the document
    :param vector: personalization vector
    :param original: a list of query word vectors
    :return: an matrix with each vector
    """
    return np.vstack([topic_doc_matrix[index].toarray(), vector, original[0], original[1]])


def query_topics(query: List[str], model_path: str, topic_doc_path, indexes) -> np.ndarray:
    """
    Takes a list of words and makes a personalization vector based on these
    :param query: list of words
    :param model_path: lda model path
    :param topic_doc_path: topic document matrix path
    :param indexes: used for comparison function
    :return: a personalization vector
    """
    lda = load_lda(model_path)
    topic_doc_matrix = sp.load_npz(topic_doc_path)
    corpus = Dictionary(load_corpus("Generated Files/corpus"))
    # q_topics = get_document_topics_from_model(query, lda, corpus)
    return make_personalization_vector_word_based(query, topic_doc_matrix, lda, indexes)


def search(size_of_adj: int, lda_path: str, vectorizer_path, topic_doc_matrix_path: str, adj_matrix_path):
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
    adj_matrix = sp.load_npz(adj_matrix_path)[:size_of_adj, :size_of_adj].toarray()
    df = pandas.read_csv("Generated Files/word2vec.csv", header=None)
    words = dict(zip(df[0], df[1]))
    queries = generate_queries(sp.load_npz(vectorizer_path)[:size_of_adj, :size_of_adj], words, 10, 4)
    for query in queries.items():
        search_vector = query_topics(query[1].split(' '), lda_path, topic_doc_matrix_path, query[0])
        doc_ranks = random_walk_with_teleport(adj_matrix, search_vector[:size_of_adj])
        print(f" Query: {query[1]} Hit: {list(doc_ranks.argsort()[:][::-1]).index(query[0])} \n "
              f"Is query dis higher than average: {np.average(search_vector[:size_of_adj]) < search_vector[:size_of_adj][query[0]]} ")


if __name__ == '__main__':
    search(2000, "LDA/model/2017_model",
           "Generated Files/count_vec_matrix.npz",
           "Generated Files/topic_doc_matrix.npz",
           "Generated Files/adj_matrix.npz")
