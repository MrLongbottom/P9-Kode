import math
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import preprocessing
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sb
from gensim import matutils
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.stats import entropy
from tqdm import tqdm
from matplotlib.cbook import boxplot_stats
from gensim.models import CoherenceModel


def fit_lda(data: csr_matrix, vocab: Dict):
    """
    Fit LDA from a scipy CSR matrix (data).
    :param data: a csr_matrix representing the vectorized words
    :param vocab: a dictionary over the words
    :return: a lda model trained on the data and vocab
    """
    print('fitting lda...')
    return LdaMulticore(matutils.Sparse2Corpus(data, documents_columns=False),
                        id2word=vocab,
                        num_topics=math.floor(math.sqrt(data.shape[1]) / 2))


def save_lda(lda: LdaModel, path: str):
    # Save model to disk.
    lda.save(path)


def load_lda(path: str):
    return LdaModel.load(path)


def create_document_topics(corpus: List[str], lda: LdaModel, filename: str) -> sp.dok_matrix:
    """
    Creates a topic_doc_matrix which describes the amount of topics in each document
    :param corpus: list of document strings
    :return: a topic_document matrix
    """
    document_topics = []
    par = partial(get_document_topics_from_model, lda=lda)
    with Pool(8) as p:
        document_topics.append(p.map(par, corpus))
    matrix = save_topic_doc_matrix(document_topics[0], lda, filename)
    return matrix


def get_document_topics_from_model(text: str, lda: LdaModel) -> Dict[int, float]:
    """
    A method used concurrently in create_document_topics
    :param lda: the lda model
    :param text: a document string
    :return: a dict with the topics in the given document based on the lda model
    """
    dictionary = Dictionary([text])
    corpus = [dictionary.doc2bow(t) for t in [text]]
    query = lda.get_document_topics(corpus, minimum_probability=0.025)
    return dict([x for x in query][0])


def save_topic_doc_matrix(document_topics: List[Dict[int, float]], lda: LdaModel, filename: str) -> sp.dok_matrix:
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
    #matrix = evaluate_doc_topic_distributions(matrix, show=True, tell=True, prune=True)
    # print once again to show improvement
    evaluate_doc_topic_distributions(matrix, show=True, tell=True, prune=False)
    sp.save_npz(filename, sp.csc_matrix(matrix))
    return matrix


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


# TODO accessing model will no longer provide the correct ids after pruning, avoid usage for now.
def evaluate_doc_topic_distributions(dt_matrix: sp.spmatrix, show: bool = True, tell: bool = True,
                                     prune: bool = False, prune_treshhold = 0.5):
    """
    Evaluate document-topic distribution matrix, involving a combination of:
    * printing statistics
    * showing boxplot
    * pruning empty docs and topics, and pruning topics that are too common
    :param dt_matrix: document-topic distribution matrix
    :param show: whether to show boxplot
    :param tell: whether to print statistics
    :param prune: whether to prune
    :param prune_treshhold: how much percentage of the doc set a topic can cover at max.
    :return: potentially pruned matrix.
    """
    sb.set_theme(style="whitegrid")
    # Topic-Doc distributions
    lens = []
    top_zeros = []
    outliers = []
    threshold = dt_matrix.shape[0] * prune_treshhold
    for i in tqdm(range(0, dt_matrix.shape[1])):
        topic = dt_matrix.getcol(i).nonzero()[0]
        lens.append(len(topic))
        if len(topic) > threshold:
            outliers.append(i)
        if len(topic) == 0:
            top_zeros.append(i)
    if tell:
        print("Topic-Doc distributions.")
        print("Minimum: " + str(min(lens)))
        print("Maximum: " + str(max(lens)))
        print("Average: " + str(np.mean(lens)))
        print("Entropy: " + str(entropy(lens, base=len(lens))))
        print("Zeros: " + str(len(top_zeros)))

    if prune:
        # TODO model cannot be accessed normally after this. Which may be a problem
        # find outlier topics based on the boxplot
        # this feature was currently discarded as running lda multiple times in a row would prune more and more.
        """
        outlier_values = [y for stat in boxplot_stats(lens) for y in stat['fliers']]
        outliers = [lens.index(x) for x in outlier_values]
        """
        # also prune topics with no documents
        outliers.extend(top_zeros)
        dt_matrix = slice_sparse_col(dt_matrix, outliers)
        tw_matrix = sp.load_npz("../Generated Files/topic_word_matrix.npz")
        tw_matrix = slice_sparse_row(tw_matrix, outliers)
        sp.save_npz("../Generated Files/topic_word_matrix.npz", tw_matrix)
    if show:
        ax = sb.boxplot(x=lens)
        plt.show()

    # Doc-Topic distributions
    lens = []
    doc_zeros = []
    for i in tqdm(range(0, dt_matrix.shape[0])):
        doc = dt_matrix.getrow(i).nonzero()[0]
        lens.append(len(doc))
        if len(doc) == 0:
            doc_zeros.append(i)
    if tell:
        print("Doc-Topic distributions.")
        print("Minimum: " + str(min(lens)))
        print("Maximum: " + str(max(lens)))
        print("Average: " + str(np.mean(lens)))
        print("Entropy: " + str(entropy(lens, base=len(lens))))
        print("Zeros: " + str(len(doc_zeros)))
    if prune:
        # TODO needs to be tested more thoroughly
        # TODO model cannot be accessed normally after this. Which may be a problem
        if len(doc_zeros) > 0:
            # Prune documents with no topic distributions
            dt_matrix = sp.csr_matrix(dt_matrix)
            dt_matrix = slice_sparse_row(dt_matrix, doc_zeros)
            remove_from_rows_from_file("../Generated Files/count_vec_matrix.npz", doc_zeros)
            remove_from_rows_from_file("../Generated Files/doc2vec.csv", doc_zeros)
            remove_from_rows_from_file("../Generated Files/doc2word.csv", doc_zeros, separator="-")
    if show:
        ax = sb.boxplot(x=lens)
        plt.show()

    return dt_matrix


def remove_from_rows_from_file (path, rows, separator=","):
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


def run_lda(path: str, cv_matrix, words, corpus, save_path):
    # fitting the lda model and saving it
    lda = fit_lda(cv_matrix, words, math.floor(math.sqrt(cv_matrix.shape[0])/2))
    save_lda(lda, path)

    # saving topic words to file
    print("creating topic words file")
    tw_matrix = save_topic_word_matrix(lda, save_path + "topic_word_matrix.npz")

    # saving document topics to file
    print("creating document topics file")
    td_matrix = create_document_topics(corpus, lda, save_path + "topic_doc_matrix.npz")

    return lda


def save_topic_word_matrix(lda: LdaModel, name: str):
    matrix = sp.csc_matrix(lda.get_topics())
    return sp.save_npz(name, matrix)


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


if __name__ == '__main__':
    # Loading data and preprocessing
    model_path = 'model_test'
    cv = sp.load_npz("../Generated Files/count_vec_matrix.npz")
    words = load_dict_file("../Generated Files/word2vec.csv")
    mini_corpus = load_dict_file("../Generated Files/doc2word.csv", separator='-')
    mini_corpus = [x[1:-1].split(', ') for x in mini_corpus.values()]
    mini_corpus = [[y[1:-1] for y in x] for x in mini_corpus]
    run_lda('model/document_model', cv, words, mini_corpus, "../Generated Files/")
