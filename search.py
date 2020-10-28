from functools import partial
from multiprocessing import Pool
from typing import Dict, List

from sklearn.preprocessing import normalize
from fast_pagerank import pagerank
import numpy as np
import pandas
import scipy.sparse as sp
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from scipy.spatial import distance
from tqdm import tqdm

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


def make_personalization_vector_word_based(word: str, topic_doc_matrix, lda):
    """
    Takes a query and transforms it into a vector based on each words topic distribution
    For each word we find its topic distribution and compare it against every other document
    using jensen shannon distance.
    :param word: a word
    :param topic_doc_matrix: topic document matrix
    :param lda: the lda model
    :return: a personalization vector (np.ndarray)
    """
    # Initialization
    p_vector = np.zeros(topic_doc_matrix.shape[0])
    vector = np.zeros(topic_doc_matrix.shape[1])
    df = pandas.read_csv("Generated Files/word2vec.csv", header=None)
    words = dict(zip(df[0], df[1]))
    topic_word_matrix = lda.get_topics()

    # Getting the word index and then getting the topic distribution for that given word
    word_index = [key for key, value in words.items() if value == word][0]
    vector += topic_word_matrix.T[word_index]

    for index, doc in enumerate(topic_doc_matrix):
        p_vector[index] = 1 - distance.jensenshannon(vector, doc.toarray()[0])
    return p_vector


def query_topics(query: List[str], model_path: str, topic_doc_path, corpus) -> np.ndarray:
    """
    Takes a list of words and makes a personalization vector based on these
    :param query: list of words
    :param model_path: lda model path
    :param topic_doc_path: topic document matrix path
    :param indexes: used for comparison function
    :return: a personalization vector
    """

    lda = load_lda(model_path)
    topic_doc_matrix = sp.load_npz(topic_doc_path)[:2000]
    p_vector = np.zeros(2000)
    for word in query:
        if word in corpus.values():
            p_vector += make_personalization_vector_word_based(word, topic_doc_matrix, lda)
        else:
            # Todo needs to be cut
            p_vector += make_personalization_vector(word, topic_doc_matrix, corpus, lda)
    return p_vector / np.linalg.norm(p_vector)


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
    adj_matrix = sp.load_npz(adj_matrix_path)[:size_of_adj, :size_of_adj]
    df = pandas.read_csv("Generated Files/word2vec.csv", header=None)
    words = dict(zip(df[0], df[1]))
    queries = generate_queries(sp.load_npz(vectorizer_path)[:size_of_adj, :size_of_adj], words, 10, 4)
    for query in queries.items():
        p_vector = query_topics(query[1].split(' '), lda_path, topic_doc_matrix_path, words)
        doc_ranks = pagerank(adj_matrix, personalize=p_vector[:size_of_adj])
        print(
            f" Query: {query[1]} PageRank Hit: {list(doc_ranks.argsort()[:][::-1]).index(query[0])} P_vector Hit: {list(p_vector.argsort()[:][::-1]).index(query[0])}")


if __name__ == '__main__':
    search(2000, "test/document_model",
           "Generated Files/tfidf_matrix.npz",
           "test/topic_doc_matrix.npz",
           "test/adj_matrix.npz")
