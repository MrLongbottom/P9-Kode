from functools import partial
from multiprocessing import Pool
from typing import Tuple, List

import numpy as np
import scipy.sparse as sp
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import preprocessing
import query_handling
import utility

cv_matrix = sp.load_npz("generated_files/count_vec_matrix.npz")
dt_matrix = sp.load_npz("generated_files/(30, 0.1, 0.1)topic_doc_matrix.npz")
tw_matrix = sp.load_npz("generated_files/(30, 0.1, 0.1)topic_word_matrix.npz")
wordfreq = cv_matrix.sum(axis=0)
doc2word = utility.load_vector_file("generated_files/doc2word.csv")
word2vec = utility.load_vector_file("generated_files/word2vec.csv")
dirichlet_smoothing = sum([len(i) for i in list(doc2word.values())]) / len(doc2word)
inverse_w2v = {v: k for k, v in word2vec.items()}


def bm25_evaluate_query(queries):
    bm25 = BM25Okapi(list(doc2word.values()))
    correct_doc_ranks = []
    all_doc_scores = []
    for doc_id, words in tqdm(queries.items()):
        scores = bm25.get_scores(words.split(' '))
        ranks = utility.rankify(dict(enumerate(scores)))
        all_doc_scores.append(ranks)
        correct_doc_ranks.append(ranks.index(doc_id))
    return correct_doc_ranks, all_doc_scores


def tfidf_evaluate_queries(queries):
    ranks = []
    for doc_id, query in queries.items():
        doc_ranks = tfidf_evaluate_query(query)
        ranks.append(doc_ranks.index(doc_id))
    return ranks


def tfidf_evaluate_query(query):
    tfidf = preprocessing.cal_tf_idf(cv_matrix)
    # model = TfidfTransformer()
    # tfidf = model.fit_transform(cv_matrix)
    re_word2vec = {v: k for k, v in word2vec.items()}
    word_vecs = []
    for word in query.split(' '):
        if word in re_word2vec:
            word_vector = tfidf.getcol(re_word2vec[word])
            word_vecs.append(word_vector.toarray())
        else:
            raise Exception("PHUCK!")
    res = np.multiply.reduce(word_vecs)
    # summing P(w|d) instead of multiplying them
    # this is often necessary to get any values
    if np.count_nonzero(res) == 0:
        word_vecs = np.stack(word_vecs)
        res = np.sum(word_vecs, axis=0)
    ranks = utility.rankify(dict(enumerate(res)))
    return ranks


def grid_lda_evaluate(query: Tuple[int, str], result_matrix: np.ndarray):
    """
    Evaluates a query based on the LDA evaluation measure presented in the paper
    :param query: query index and a string
    :param result_matrix: dt matrix * tw matrix
    :return: returns the ranks and personalization vector
    """
    document_index = query[0]
    word_indexes = [inverse_w2v[x] for x in query[1].split(' ')]

    value = []
    for word_index in word_indexes:
        value.append(result_matrix[:, word_index])
    p_vec = np.multiply.reduce(value)
    res = utility.rankify(dict(enumerate(p_vec))).index(document_index)
    return res, p_vec


def lda_evaluate_word_doc(document_index, word_index):
    """
    The LDA evaluates a document against a word and returns the score
    :param document_index: document
    :param word_index: word
    :return: a score
    """
    word_topics = tw_matrix.getcol(word_index)
    doc_topics = dt_matrix[document_index].T
    score = word_topics.multiply(doc_topics).sum()
    return score


def lm_evaluate_word_doc(document_index, word_index):
    """
    The language model evaluates a document against a word and returns the score
    :param document_index: document
    :param word_index: word
    :return: a score
    """
    N_d = len(doc2word[document_index])
    tf = cv_matrix[document_index, word_index]
    w_freq_in_D = wordfreq[word_index].data[0]
    number_of_word_tokens = len(word2vec)
    score = ((N_d / (N_d + dirichlet_smoothing)) * (tf / N_d)) + \
            ((1 - (N_d / (N_d + dirichlet_smoothing))) * (
                    w_freq_in_D / number_of_word_tokens))
    return score


def evaluate_document_query(queries, dt_matrix, tw_matrix, evaluation_function):
    """

    :param queries: queries
    :param dt_matrix: document topic matrix
    :param tw_matrix: topic word matrix
    :param evaluation_function: the given evaluation function
    :return: evaluation results
    """
    result_matrix = np.matmul(dt_matrix.A, tw_matrix.A)
    results = []
    for query in tqdm(queries):
        res, p_vec = evaluation_function(query, result_matrix)
        results.append(res)
    return results


def evaluate_query(function, query_index, query_words, tell=False):
    """
    Evaluating a query based on a function given and the query
    which consists of query index and words
    :param function: the evaluation function you want to use
    :param query_index: the query's document index
    :param query_words: the words in the query
    :param tell: do you want to print top 3 and the words after it has finished
    :return: the index of the query in the ranked list and the list it self.
    """
    lst = {}
    with Pool(processes=8) as p:
        max_ = len(doc2word)
        with tqdm(total=max_) as pbar:
            for i, score in enumerate(
                    p.imap(partial(evaluate_query_doc, function, query_words), range(0, max_))):
                lst[i] = score
                pbar.update()

    sorted_list = utility.rankify(lst)
    if tell:
        print(f"query: {query_words}")
        print(f"index of document: {sorted_list.index(query_index)}")
        print(f"number 1 index: {sorted_list[0]} number 1: words {doc2word[sorted_list[0]]}\n")
        print(f"number 2 index: {sorted_list[1]} number 2: words {doc2word[sorted_list[1]]}\n")
        print(f"number 3 index: {sorted_list[2]} number 3: words {doc2word[sorted_list[2]]}\n")
        print(f"real document: {doc2word[query_index]}")
    return sorted_list.index(query_index), lst


def evaluate_query_doc(function, query: List[str], document_index: int):
    """
    Evaluate a query based on a function and document index
    :param function: the evaluation function
    :param query: the list of query words
    :param document_index: the index of the document
    :return: the product of the evaluate function
    """
    p_wd = []
    for word in query:
        word_index = inverse_w2v[word]
        p_wd.append(function(document_index, word_index))
    return np.prod(p_wd)


if __name__ == '__main__':
    queries = query_handling.generate_document_queries(cv_matrix, word2vec, 100, 4, 4)
    ranks = evaluate_document_query(queries.items(), dt_matrix, tw_matrix, grid_lda_evaluate)
    print(ranks)
