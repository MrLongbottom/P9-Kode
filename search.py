from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
from gensim.corpora import Dictionary
from tqdm import tqdm

from lda import load_lda, get_document_topics_from_model, load_corpus
from cluster_random_walk import cluster_page_rank
from preprocessing import preprocess_query, generate_queries, load_vector_file


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


def load_doc_2_word(path, seperator=','):
    with open(path, 'r') as file:
        dictionary = {}
        for line in file.readlines():
            key = int(line.split(seperator)[0])
            value = line.replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(seperator)[1:][
                0].split(',')
            dictionary[key] = value
    return dictionary


def search(query: str, size_of_adj: int, lda_path: str, topic_doc_matrix_path: str) -> np.ndarray:
    search_vector = query_topics(query, lda_path, topic_doc_matrix_path)
    adj_matrix = sp.load_npz("Generated Files/full_matrix.npz")[:size_of_adj, :size_of_adj]
    return cluster_page_rank(adj_matrix, search_vector[:size_of_adj])


def language_model(query: List[str], document_index: int):
    p_wd = []
    document = doc2word[document_index]
    dirichlet_prior = sum([len(i) for i in list(doc2word.values())]) / len(doc2word)

    for word in query:
        word_index = inverse_w2v[word]
        N_d = len(document)
        tf = count_vectorizer[document_index, word_index]
        w_freq_in_D = sum(count_vectorizer[:, word_index])
        number_of_word_tokens = len(word2vec)
        p_wd.append(
            (N_d / (N_d + dirichlet_prior)) *
            (tf / N_d) +
            (1 - (N_d / (N_d + dirichlet_prior))) *
            (w_freq_in_D / number_of_word_tokens))
    return np.prod(p_wd)


if __name__ == '__main__':

    tf_idf_vec = sp.load_npz("Generated Files/tfidf_matrix.npz")
    count_vectorizer = sp.load_npz("Generated Files/count_vec_matrix.npz")

    doc2word = load_doc_2_word("Generated Files/doc2word.csv", '-')
    word2vec = load_vector_file("Generated Files/word2vec.csv")
    inverse_w2v = {v: k for k, v in word2vec.items()}

    queries = generate_queries(tf_idf_vec, word2vec, 10, 4)
    query_words = list(queries.items())[0][1].split(' ')
    query_index = list(queries.items())[0][0]

    lst = []
    with Pool(processes=8) as p:
        max_ = count_vectorizer.shape[0]
        with tqdm(total=max_) as pbar:
            for i, score in enumerate(p.imap_unordered(partial(language_model, query_words), range(0, max_))):
                lst.append((i, score))
                pbar.update()
    sorted_list = list(dict(sorted(lst, key=lambda x: x[1], reverse=True)).keys())
    print(f"index of document: {sorted_list.index(query_index)}")
