import numpy as np
import scipy.sparse as sp

import utility

cv_matrix = sp.load_npz("Generated Files/count_vec_matrix.npz")
dt_matrix = sp.load_npz("Generated Files/topic_doc_matrix.npz")
tw_matrix = sp.load_npz("Generated Files/topic_word_matrix.npz")
wordfreq = cv_matrix.sum(axis=0)
doc2word = utility.load_vector_file("Generated Files/doc2word.csv")
word2vec = utility.load_vector_file("Generated Files/word2vec.csv")
dirichlet_smoothing = sum([len(i) for i in list(doc2word.values())]) / len(doc2word)


def lda_evaluate_word_doc(document_index, word_index, matrices=None):
    """
    The LDA evaluates a document against a word and returns the score
    :param matrices: the two matrices dt, tw
    :param document_index: document
    :param word_index: word
    :return: a score
    """
    if matrices:
        dt_matrix = matrices[0]
        tw_matrix = matrices[1]
    word_topics = tw_matrix.getcol(word_index)
    doc_topics = dt_matrix[document_index].T
    score = word_topics.multiply(doc_topics).sum()
    return score


def lm_evaluate_word_doc(document_index, word_index, matrices=None):
    """
    The language model evaluates a document against a word and returns the score
    :param matrices: the two matrices dt, tw
    :param document_index: document
    :param word_index: word
    :return: a score
    """
    N_d = len(doc2word[document_index])
    tf = cv_matrix[document_index, word_index]
    w_freq_in_D = np.array(wordfreq.T[word_index])[0][0]
    number_of_word_tokens = len(word2vec)
    score = ((N_d / (N_d + dirichlet_smoothing)) * (tf / N_d)) + \
            ((1 - (N_d / (N_d + dirichlet_smoothing))) * (
                    w_freq_in_D / number_of_word_tokens))
    return score


def lm_lda_combo_evaluate_word_doc(document_index, word_index, matrices=None):
    """
    Combines the two score functions from LDA and LM
    :param matrices: The two matrices dt, tw
    :param document_index: document
    :param word_index: word
    :return: a score
    """
    if matrices:
        dt_matrix = matrices[0]
        tw_matrix = matrices[1]
    lm_score = lm_evaluate_word_doc(document_index, word_index, matrices)
    lda_score = lda_evaluate_word_doc(document_index, word_index, matrices)
    return lda_score * lm_score
