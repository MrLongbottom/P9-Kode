import os
from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
import scipy.sparse as sp
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize
from tqdm import tqdm

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


def tfidf_evaluate_query(query: List[str]):
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


def lda_evaluate(query: List[str]):
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
    prob = []
    for word_index in word_indexes:
        word_probability = []
        for document_index in range(dt_matrix.shape[0]):
            N_d = len(doc2word[document_index])
            tf = cv_matrix[document_index, word_index]
            w_freq_in_D = wordfreq[word_index]
            number_of_word_tokens = len(word2vec)
            score = ((N_d / (N_d + dirichlet_smoothing)) * (tf / N_d)) + \
                    ((1 - (N_d / (N_d + dirichlet_smoothing))) * (
                            w_freq_in_D / number_of_word_tokens))
            word_probability.append(score)
        prob.append(np.array(word_probability))
    return np.multiply.reduce(prob)


def hit_point():
    """
    Calculates the hit accuracy for the first 4 queries
    :return: the average index of where it hit
    """
    hits = []
    for i in range(4):
        hit = []
        ranks = [utility.rankify(dict(enumerate(x))) for x in matrices[i]]
        for query_n, (answer, _) in enumerate(queries[i]):
            # GTP is answer
            hit.append(ranks[query_n].index(answer) + 1)
        hits.append(np.mean(hit))
        print(np.mean(hit))
    return hits


def precision_at_x(X, matrices):
    """
    Calculates precision at X
    :param matrices: 4 ndarrays of document queries + 4 ndarrays of documents
    :param X: int
    :return: precision
    """
    precisions = []
    for i in range(8):
        precision = []
        ranks = [utility.rankify(dict(enumerate(x))) for x in matrices[i]]
        if i < 4:
            for query_n, (answer, _) in enumerate(queries[i]):
                # GTP is answer
                if ranks[query_n].index(answer) < X:
                    precision.append(1 / X)
        else:
            with Pool(processes=8) as p:
                max_ = len(list(enumerate(queries[i])))
                with tqdm(total=max_) as pbar:
                    for _, score in enumerate(
                            p.starmap(partial(precision_function, ranks, X), list(enumerate(queries[i])))):
                        precision.append(score)
                        pbar.update()
        precisions.append(np.mean(precision))
        print(np.mean(precision))
    return precisions


def mean_average_precision(matrices):
    """
    Calculates the documents queries for the first 4 iterations and
    then it calculates the topic queries for the last 4 iterations
    :param matrices: the 8 query sets
    :return: mean average precision
    """
    MAP = []
    for i in range(8):  # 8 because there are 8 sets of queries of different lengths
        AP = []
        ranks = [utility.rankify(dict(enumerate(x))) for x in matrices[i]]
        if i < 4:
            for query_n, (answer, _) in enumerate(queries[i]):
                # GTP is answer
                AP.append(1 / (ranks[query_n].index(answer) + 1))
        else:
            with Pool(processes=8) as p:
                max_ = len(list(enumerate(queries[i])))
                with tqdm(total=max_) as pbar:
                    for _, score in enumerate(p.starmap(partial(mean_average_precision_inner_function, ranks),
                                                        list(enumerate(queries[i])))):
                        AP.append(score)
                        pbar.update()
        MAP.append(np.mean(AP))
        print(np.mean(AP))
    return MAP


def mean_average_precision_inner_function(ranks, query_n, answer):
    """
    This function is used within the mean_average_precision function and
    the gtp within this function is ground truth positives.
    :param ranks: the ranking of the documents
    :param query_n: the index of the query we are working with
    :param answer: what the index of the document we want.
    :return: mean average precision
    """
    topic = dt_matrix.getcol(answer[0]).toarray()
    threshold = topic.mean()
    gtp_ids = np.nonzero(np.where(topic < threshold, 0, topic))[0]
    gtp_ranks = [ranks[query_n].index(gtp_id) for gtp_id in gtp_ids]
    gtp_ranks.sort()
    precision = [(i + 1) / (gtp_ranks[i] + 1) for i in range(len(gtp_ranks))]
    return np.mean(precision)


def precision_function(ranks, X, query_n, answer):
    topic = dt_matrix.getcol(answer[0]).toarray()
    threshold = topic.mean()
    gtp_ids = np.nonzero(np.where(topic < threshold, 0, topic))[0]
    gtp_in_N = [x for x in ranks[query_n][:X] if x in gtp_ids]
    return len(gtp_in_N) / X


if __name__ == '__main__':
    paths = ["queries/" + x for x in os.listdir("queries/")]
    paths.sort()
    paths = paths[4:12]
    doc_queries = [utility.load_vector_file(x) for x in paths[:4]]
    queries = [[(x, y) for x, y in q.items()] for q in doc_queries]
    queries.extend([utility.load_vecter_file_nonunique(x) for x in paths[4:]])
    # matrices = []
    # for queryset in queries:
    #    matrices.append(np.array(query_handling.evaluate_queries(queryset, bm25_evaluate_query)))
    # # save matrix
    # np.save("bm25_evaluate_matrices", matrices)

    # load matrix
    model1 = list(np.load("data/pr_matrix.npy"))
    model1 = ([np.vstack([np.array(model1), ] * 80), ] * 8)
    model2 = list(np.load("lda_evaluate_matrices.npy"))
    model3 = list(np.load("bm25_evaluate_matrices.npy"))
    matrices = [np.add(np.add(normalize(a, norm="l1"), normalize(b, norm='l1')), normalize(c, norm='l1')) for c, b, a in
                zip(model1, model2, model3)]

    pre10 = precision_at_x(10, matrices)
    utility.save_vector_file("Generated Files/bm25_pre_10", pre10)
