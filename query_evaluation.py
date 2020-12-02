import os
from typing import List

import numpy as np
import scipy.sparse as sp
from rank_bm25 import BM25Okapi

import preprocessing
import query_handling
import utility

cv_matrix = sp.load_npz("Generated Files/count_vec_matrix.npz")
dt_matrix = sp.load_npz("Generated Files/(30, 0.1, 0.1)topic_doc_matrix.npz")
tw_matrix = sp.load_npz("Generated Files/(30, 0.1, 0.1)topic_word_matrix.npz")
wordfreq = np.array(cv_matrix.sum(axis=0))[0]
doc2word = utility.load_vector_file("Generated Files/doc2word.csv")
word2vec = utility.load_vector_file("Generated Files/word2vec.csv")
dirichlet_smoothing = sum([len(i) for i in list(doc2word.values())]) / len(doc2word)
inverse_w2v = {v: k for k, v in word2vec.items()}
result_matrix = np.matmul(dt_matrix.A, tw_matrix.A)
bm25 = BM25Okapi(list(doc2word.values()))


def bm25_evaluate_query(query: List[str]):
    return bm25.get_scores(query)


def tfidf_evaluate_query(query):
    tfidf = preprocessing.cal_tf_idf(cv_matrix)
    # model = TfidfTransformer()
    # tfidf = model.fit_transform(cv_matrix)
    word_vecs = []
    for word in query:
        if word in inverse_w2v:
            word_vector = tfidf.getcol(inverse_w2v[word])
            word_vecs.append(word_vector.toarray())
        else:
            raise Exception("PHUCK!")
    res = np.multiply.reduce(word_vecs)
    # summing P(w|d) instead of multiplying them
    # this is often necessary to get any values
    if np.count_nonzero(res) == 0:
        word_vecs = np.stack(word_vecs)
        res = np.sum(word_vecs, axis=0)
    return res


def lda_evaluate(query: List[str], result_matrix: np.ndarray):
    word_indexes = [inverse_w2v[x] for x in query]

    value = []
    for word_index in word_indexes:
        value.append(result_matrix[:, word_index])
    p_vec = np.multiply.reduce(value)
    return p_vec


def lm_evaluate_query(query: List[str]):
    """
    The language model evaluates a document against a word and returns the score
    :param query: a list of query words
    :return: a score
    """
    word_indexes = [inverse_w2v[x] for x in query]
    word_probability = []
    for word_index in word_indexes:
        for document_index in range(dt_matrix.shape[0]):
            N_d = len(doc2word[document_index])
            tf = cv_matrix[document_index, word_index]
            w_freq_in_D = wordfreq[word_index]
            number_of_word_tokens = len(word2vec)
            score = ((N_d / (N_d + dirichlet_smoothing)) * (tf / N_d)) + \
                    ((1 - (N_d / (N_d + dirichlet_smoothing))) * (
                            w_freq_in_D / number_of_word_tokens))
            word_probability.append(score)
    return np.prod(word_probability)


def precision_at_X(query_value_matrix, query_answers, limit):
    hit = 0
    for i, query_values in enumerate(query_value_matrix):
        topX = query_values.argsort()[-limit:][::-1]
        if query_answers in topX:
            hit += 1
    return hit


if __name__ == '__main__':
    paths = ["queries/" + x for x in os.listdir("queries/")]
    queries = [utility.load_vector_file(x) for x in paths]
    matrices = []
    for queryset in queries:
        matrices.append(np.array(query_handling.evaluate_queries(queryset.items(), lm_evaluate_query)))
    # save matrix
    np.save("lm_evaluate_matrices", matrices)

    # load matrix
    # matrix = list(np.load("lda_evaluate_matrices.npy"))
