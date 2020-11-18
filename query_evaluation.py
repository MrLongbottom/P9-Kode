import numpy as np
import scipy.sparse as sp
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import preprocessing
import utility

cv_matrix = sp.load_npz("Generated Files/count_vec_matrix.npz")
dt_matrix = sp.load_npz("Generated Files/topic_doc_matrix.npz")
tw_matrix = sp.load_npz("Generated Files/topic_word_matrix.npz")
wordfreq = cv_matrix.sum(axis=0)
doc2word = utility.load_vector_file("Generated Files/doc2word.csv")
word2vec = utility.load_vector_file("Generated Files/word2vec.csv")
dirichlet_smoothing = sum([len(i) for i in list(doc2word.values())]) / len(doc2word)


def tfidf_evaluate_queries(queries):
    ranks = {}
    for doc_id, query in queries.items():
        ranks[query] = tfidf_evaluate_query(query).get(doc_id)
    return ranks


def tfidf_evaluate_query(query):
    tfidf = preprocessing.cal_tf_idf(cv_matrix)
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
    test = tfidf_evaluate_query("hej fodbold rejse")
    print()
