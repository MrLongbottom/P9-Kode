import numpy as np
import scipy.sparse as sp
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import query_handling
import utility

cv_matrix = sp.load_npz("Generated Files/count_vec_matrix.npz")
dt_matrix = sp.load_npz("Generated Files/topic_doc_matrix.npz")
tw_matrix = sp.load_npz("Generated Files/topic_word_matrix.npz")
wordfreq = cv_matrix.sum(axis=0)
doc2word = utility.load_vector_file("Generated Files/doc2word.csv")
word2vec = utility.load_vector_file("Generated Files/word2vec.csv")
dirichlet_smoothing = sum([len(i) for i in list(doc2word.values())]) / len(doc2word)


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


def lm_lda_combo_evaluate_word_doc(document_index, word_index):
    """
    Combines the two score functions from LDA and LM
    :param document_index: document
    :param word_index: word
    :return: a score
    """
    lm_score = lm_evaluate_word_doc(document_index, word_index)
    lda_score = lda_evaluate_word_doc(document_index, word_index)
    return lda_score * lm_score


if __name__ == '__main__':
    queries = query_handling.generate_document_queries(cv_matrix, word2vec, 100, 4, 4)
    rank, _ = bm25_evaluate_query(queries)
    print(rank)
