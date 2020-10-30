import itertools
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple

from sklearn.preprocessing import normalize
from fast_pagerank import pagerank
import numpy as np
import pandas as pd
import pandas
import scipy.sparse as sp
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from scipy.spatial import distance
from tqdm import tqdm

from lda import load_lda, get_document_topics_from_model, load_corpus, get_document_topics_from_model_from_texts
from preprocessing import generate_queries, load_vector_file
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


def load_doc_2_word(path, seperator=','):
    with open(path, 'r') as file:
        dictionary = {}
        for line in file.readlines():
            key = int(line.split(seperator)[0])
            value = line.replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(seperator)[1:][0].split(',')
            dictionary[key] = value
    return dictionary


def query_expansion(query, window_size: int = 1, n_top_word: int = 10):
    documents = load_doc_2_word("Generated Files/doc2word.csv", '-')
    doc_id = query[0]
    result = []
    words = query[1]
    for word in words.split(' '):
        expanded_query = {}
        # append original word to query
        document_ids = [ids for ids, values in documents.items() if word in values]
        for new_id_doc in document_ids:
            # add window size neighboring words
            document = documents[new_id_doc]
            word_index = document.index(word)
            if word_index != len(document):
                before_word = document[word_index - 1]
                after_word = document[word_index + 1]
                expanded_query[before_word] = expanded_query.get(before_word, 0) + 1
                expanded_query[after_word] = expanded_query.get(after_word, 0) + 1
        sorted_query_words = list(dict(sorted(expanded_query.items(), key=lambda x: x[1], reverse=True)).keys())
        result.append(sorted_query_words[:n_top_word])
    result.append(words.split(' '))
    return list(set(itertools.chain.from_iterable(result)))


if __name__ == '__main__':
    vectorizer = sp.load_npz("Generated Files/tfidf_matrix.npz")
    words = load_vector_file("Generated Files/word2vec.csv")
    dictionary = Dictionary([words.values()])
    queries = generate_queries(vectorizer, words, 10, 4)
    expanded_queries = []

    for query in tqdm(list(queries.items())):
        expanded_queries.append(query_expansion(query, 1, 2))

    lda = load_lda("LDA/model/document_model")
    topic_distributions = []
    for exp_query in expanded_queries:
        topic_distributions.append(get_document_topics_from_model_from_texts(expanded_queries, lda, dictionary, 0.025))

    for query, topic_dis in zip(expanded_queries, topic_distributions):
        print(f"Topic distribution: {topic_dis}")
