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
    # matrix = evaluate_doc_topic_distributions(matrix, show=True, tell=True, prune=True)
    # print once again to show improvement
    # evaluate_doc_topic_distributions(matrix, show=True, tell=True, prune=False)
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
def evaluate_distribution_matrix(dis_matrix: sp.spmatrix, show: bool = True, tell: bool = True,
                                 name1: str = "A", name2: str = "B"):
    """
    Evaluate document-topic distribution matrix, involving a combination of:
    * printing statistics
    * showing boxplot
    * pruning empty docs and topics, and pruning topics that are too common
    :param dis_matrix: document-topic distribution matrix
    :param show: whether to show boxplot
    :param tell: whether to print statistics
    :param prune: whether to prune
    :param prune_treshhold: how much percentage of the doc set a topic can cover at max.
    :return: potentially pruned matrix.
    """
    sb.set_theme(style="whitegrid")
    # loop over A-B distribution, then B-A distribution
    for ab in range(0, 2):
        zeros = []
        empties = []
        avgs = []
        maxs = []
        mins = []
        medians = []
        entropies = []
        max_loop = 1 if ab == 0 else 0
        for i in tqdm(range(0, dis_matrix.shape[max_loop])):
            vec = dis_matrix.getcol(i) if ab == 0 else dis_matrix.getrow(i)
            non_vec = vec.nonzero()[ab]
            zeros.append(vec.shape[ab] - len(non_vec))
            avgs.append(vec.mean())
            maxs.append(vec.max())
            mins.append(vec.min())
            medians.append(np.median(vec.toarray()))
            if len(non_vec) == 0:
                empties.append(i)
            else:
                # TODO check if this should always be transposed (i think its only for one of the two ab's)
                entropies.append(entropy(vec.toarray().T[0], base=vec.shape[ab]))
        return_stats = []
        if tell:
            if ab == 0:
                print(f"{name1}-{name2} Distributions.")
            else:
                print(f"{name2}-{name1} distributions.")
            print(f"{len(empties)} empty vectors")
        stats = {"Number of zeros": zeros, "Minimums": mins, "Maximums": maxs, "Averages": avgs, "Medians": medians,
                 "Entropies": entropies}
        for name, stat in stats.items():
            return_stats.extend(stats_of_list(stat, name=name, tell=tell))

        #if show:
        #    ax = sb.boxplot(x=zeros)
        #    plt.show()
        return return_stats


def stats_of_list(list, name: str = "List", tell: bool = True):
    zeros = (len(list) - np.count_nonzero(list))/len(list)
    mini = min(list)
    maxi = max(list)
    avg = np.mean(list)
    medi = np.median(list)
    entro = 1 if np.isnan(entropy(list, base=len(list))) else entropy(list, base=len(list))
    if tell:
        print(f"{name} Statistics.")
        print(f"Zeros percentage in {name}: {zeros}")
        print(f"Minimum {name}: {mini}")
        print(f"Maximum {name}: {maxi}")
        print(f"Average {name}: {avg}")
        print(f"Median {name}: {medi}")
        print(f"Entropy {name}: {entro}")
        print()
    return [zeros, mini, maxi, avg, medi, entro]


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


def run_lda(path: str, cv_matrix, words, corpus, save_path):
    # fitting the lda model and saving it
    lda = fit_lda(cv_matrix, words)
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
    # Evaluate
    td_matrix = sp.load_npz("../Generated Files/topic_doc_matrix.npz")
    tw_matrix = sp.load_npz("../Generated Files/topic_word_matrix.npz")
    evaluate_distribution_matrix(td_matrix, name1="Topic", name2="Document")
    evaluate_distribution_matrix(tw_matrix, name1="Word", name2="Topic")
