import json
import math
import itertools
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import time
import preprocessing
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sb
from gensim import matutils
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models import LdaModel, LdaMulticore
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.stats import entropy
from tqdm import tqdm
from matplotlib.cbook import boxplot_stats
from gensim.models import CoherenceModel

import preprocessing
import evaluate
import utility


def fit_lda(data: csr_matrix, vocab: Dict, K: int, alpha: float = None, eta: float = None):
    """
    Fit LDA from a scipy CSR matrix (data).
    :param data: a csr_matrix representing the vectorized words
    :param vocab: a dictionary over the words
    :param K: number of topics
    :param alpha: the alpha prior weight of topics in documents
    :param eta: the eta prior weight of words in topics
    :return: a lda model trained on the data and vocab
    """
    print('fitting lda...')
    if alpha is None or eta is None:
        return LdaMulticore(matutils.Sparse2Corpus(data, documents_columns=False),
                            id2word=vocab,
                            num_topics=K)
    else:
        return LdaMulticore(matutils.Sparse2Corpus(data, documents_columns=False),
                            id2word=vocab,
                            num_topics=K,
                            alpha=alpha,
                            eta=eta)


def save_lda(lda: LdaModel, path: str):
    # Save model to disk.
    lda.save(path)


def load_lda(path: str):
    return LdaModel.load(path)


def create_document_topics(corpus: List[str], lda: LdaModel, filename: str, dictionary: Dictionary,
                           K) -> sp.dok_matrix:
    """
    Creates a topic_doc_matrix which describes the amount of topics in each document
    :param corpus: list of document strings
    :return: a topic_document matrix
    """
    document_topics = []
    par = partial(get_document_topics_from_model, lda=lda, dictionary=dictionary, K=K)
    with Pool(8) as p:
        document_topics.append(p.map(par, corpus))
    matrix = save_topic_doc_matrix(document_topics[0], lda, filename)
    return matrix


def load_corpus(name: str):
    with open(name, 'r', encoding='utf8') as json_file:
        corpus = json.loads(json_file.read())
    return corpus


def get_document_topics_from_model(text: str, lda: LdaModel, dictionary: Dictionary, K) -> Dict[int, float]:
    """
    A method used concurrently in create_document_topics
    :param lda: the lda model
    :param text: a document string
    :param dictionary: the dictionary over the whole document
    :return: a dict with the topics in the given document based on the lda model
    """
    corpus = [dictionary.doc2bow(t) for t in [text]]
    query = lda.get_document_topics(corpus, minimum_probability=1/K)
    return dict([x for x in query][0])


def save_topic_doc_matrix(document_topics: List[Dict[int, float]], lda: LdaModel, filename: str) -> sp.csc_matrix:
    """
    Saves the document topics (list of dicts) in a matrix
    :param document_topics: list of dicts
    :param lda: the lda model
    :param file_name: path of file to save
    :return: a matrix (scipy)
    """
    matrix = sp.dok_matrix((len(document_topics), lda.num_topics))
    for index, dictionary in tqdm(enumerate(document_topics)):
        for dict_key, dict_value in dictionary.items():
            matrix[index, dict_key] = dict_value
    # matrix = evaluate_doc_topic_distributions(matrix, show=True, tell=True, prune=True)
    # print once again to show improvement
    # evaluate_doc_topic_distributions(matrix, show=True, tell=True, prune=False)
    sp.save_npz(filename, sp.csc_matrix(matrix))
    return sp.csc_matrix(matrix)


def word_cloud(corpus):
    # Import the wordcloud library
    from wordcloud import WordCloud
    # Join the different processed titles together.
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(str(corpus))
    # Visualize the word cloud
    wordcloud.to_image().show()


def remove_from_rows_from_file(path, rows, separator=","):
    if path[-4:] == ".csv":
        doc = load_dict_file(path, separator=separator)
        doc = [x for x in doc.values() if x not in rows]
        preprocessing.save_vector_file(path, doc, seperator=separator)
    elif path[-4:] == ".npz":
        matrix = sp.load_npz(path)
        s = matrix.shape
        matrix = slice_sparse_row(matrix, rows)
        s2 = matrix.shape
        print()


def slice_sparse_col(matrix: sp.csc_matrix, cols: List[int]):
    """
    Remove some columns from a sparse matrix.
    :param matrix: CSC matrix.
    :param cols: list of column numbers to be removed.
    :return: modified matrix without the specified columns.
    """
    cols.sort()
    ms = []
    prev = -1
    for c in cols:
        # add slices of the matrix, skipping column c
        ms.append(matrix[:, prev + 1:c - 1])
        prev = c
    ms.append(matrix[:, prev + 1:])
    # combine matrix slices
    return sp.hstack(ms)


def slice_sparse_row(matrix: sp.csr_matrix, rows: List[int]):
    """
    Remove some rows from a sparse matrix.
    :param matrix: CSR matrix.
    :param rows: list of row numbers to be removed.
    :return: modified matrix without the specified rows.
    """
    rows.sort()
    ms = []
    prev = -1
    for r in rows:
        # add slices of the matrix, skipping row r
        ms.append(matrix[prev + 1:r - 1, :])
        prev = r
    ms.append(matrix[prev + 1:, :])
    # combine matrix slices
    return sp.vstack(ms)


def run_lda(path: str, cv_matrix, words, corpus, dictionary, save_path, param_combination: tuple):
    # fitting the lda model and saving it
    lda = fit_lda(cv_matrix, words, param_combination[0], param_combination[1], param_combination[2])
    save_lda(lda, path)

    # saving topic words to file
    print("creating topic words file")
    tw_matrix = save_topic_word_matrix(lda,
                                       save_path + str(param_combination ) + "topic_word_matrix.npz")

    # saving document topics to file
    print("creating document topics file")
    dt_matrix = create_document_topics(corpus, lda,
                                       save_path + str(param_combination) + "topic_doc_matrix.npz",
                                       dictionary, param_combination[0])

    return lda, dt_matrix, tw_matrix


def save_topic_word_matrix(lda: LdaModel, name: str):
    matrix = lda.get_topics()
    threshold = 1/matrix.shape[1]
    matrix = np.where(matrix < threshold, 0, matrix)
    matrix = sp.csr_matrix(matrix)
    sp.save_npz(name, matrix)
    return matrix


def get_topic_word_matrix(lda: LdaModel) -> np.ndarray:
    return lda.get_topics()


def load_dict_file(path, separator=','):
    csv_reader = pd.read_csv(path, header=None, encoding='unicode_escape', sep=separator)
    dic = dict(csv_reader.values.tolist())
    return dic


def print_topic_words(id: int, lda_model: LdaModel):
    return dict(lda_model.show_topics(lda_model.num_topics))[id]


def coherence_score(lda: LdaModel, texts, id2word, measure: str = 'c_v'):
    coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence=measure)
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


def compute_coherence_values(cv_matrix, words, dictionary, texts, limit, start=2, step=10):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []

    for num_topics in tqdm(range(start, limit, step)):
        model = fit_lda(cv_matrix, words, num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def compute_coherence_values_k_and_priors(cv_matrix, words, dictionary, texts,
                                          Ks: List[int], alphas: List[float], etas: List[float],
                                          thresholds: List[float], evaluation: bool = True):
    """
    Compute c_v coherence for various number of topics and priors

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    Ks : A list of K values to apply
    alphas : A list of alpha values to apply
    etas : A list of eta values to apply

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics and priors
    """
    coherence_values = []
    model_list = []
    dt_eval_results = []
    tw_eval_results = []
    completed_dt_evals = []
    mini_corpus = load_mini_corpus()

    test_combinations = list(itertools.product(Ks, alphas, etas, thresholds))
    for combination in tqdm(test_combinations):
        model = run_lda('LDA/model/' + str(combination[0:3]) + 'document_model',
                        cv_matrix,
                        words,
                        mini_corpus,
                        Dictionary(mini_corpus),
                        "Generated Files/",
                        combination[0:3])
        model_list.append(model)

        # Evaluation
        if evaluation:
            dtMatrix = sp.load_npz("Generated Files/" + str(combination[0:3] + (0.025,)) + "topic_doc_matrix.npz")
            twMatrix = sp.load_npz("Generated Files/" + str(combination) + "topic_word_matrix.npz")
            dtPath = "Generated Files/Evaluate/dt" + str(combination[0:3] + (0.025,))
            twPath = "Generated Files/Evaluate/tw" + str(combination)
            if combination[0:3] not in completed_dt_evals:
                completed_dt_evals.append(combination[0:3])
                dt_eval_results.append(
                    evaluate.evaluate_distribution_matrix(dtMatrix, column_name="topic", row_name="document",
                                                          save_path=None, show=False, tell=False))
            tw_eval_results.append(
                evaluate.evaluate_distribution_matrix(twMatrix, column_name="word", row_name="topic", save_path=None,
                                                      show=False, tell=False))

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values, dt_eval_results, tw_eval_results


def load_mini_corpus():
    mini_corpus = load_dict_file("Generated Files/doc2word.csv", separator='-')
    mini_corpus = [x[1:-1].split(', ') for x in mini_corpus.values()]
    mini_corpus = [[y[1:-1] for y in x] for x in mini_corpus]
    return mini_corpus


if __name__ == '__main__':
    # Loading data and preprocessing
    model_path = 'LDA/model/document_model'
    cv = sp.load_npz("Generated Files/count_vec_matrix.npz")
    words = load_dict_file("Generated Files/word2vec.csv")
    mini_corpus = utility.load_vector_file("Generated Files/doc2word.csv").values()
    K = math.floor(math.sqrt(cv.shape[0]) / 2)
    run_lda(model_path,
            cv,
            words,
            mini_corpus,
            Dictionary(mini_corpus),
            "Generated Files/",
            (K, None, None))

    # lda = load_lda("model/document_model")
    # corpus = load_corpus("../Generated Files/corpus")
    # coherence_score(lda, corpus, Dictionary(corpus))
