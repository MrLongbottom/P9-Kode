import math
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sb
from gensim import matutils
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from gensim.test.utils import datapath
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.stats import entropy
from tqdm import tqdm


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


def create_document_topics(corpus: List[str], lda: LdaModel) -> sp.dok_matrix:
    """
    Creates a topic_doc_matrix which describes the amount of topics in each document
    :param corpus: list of document strings
    :return: a topic_document matrix
    """
    document_topics = []
    par = partial(get_document_topics_from_model, lda=lda)
    with Pool(8) as p:
        document_topics.append(p.map(par, corpus))
    matrix = save_topic_doc_matrix(document_topics[0], lda)
    return matrix


def get_document_topics_from_model(text: str, lda: LdaModel) -> Dict[int, float]:
    """
    A method used concurrently in create_document_topics
    :param lda: the lda model
    :param text: a document string
    :return: a dict with the topics in the given document based on the lda model
    """
    tokenized_text = [text.split(' ')]
    dictionary = Dictionary(tokenized_text)
    corpus = [dictionary.doc2bow(t) for t in tokenized_text]
    query = lda.get_document_topics(corpus, minimum_probability=0.025)
    return dict([x for x in query][0])


def save_topic_doc_matrix(document_topics: List[Dict[int, float]], lda: LdaModel) -> sp.dok_matrix:
    """
    Saves the document topics (list of dicts) in a matrix
    :param document_topics: list of dicts
    :return: a matrix (scipy)
    """
    matrix = sp.dok_matrix((len(document_topics), lda.num_topics))
    for index, dictionary in tqdm(enumerate(document_topics)):
        for dict_key, dict_value in dictionary.items():
            matrix[index, dict_key] = dict_value
    sp.save_npz("Generated Files/test_topic_doc_matrix", sp.csc_matrix(matrix))
    return matrix


def word_cloud(corpus):
    # Import the wordcloud library
    from wordcloud import WordCloud
    # Join the different processed titles together.
    long_string = ','.join(corpus)
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image().show()


def evaluate_doc_topic_distributions(dtm):
    lens = []
    zeros = 0
    for i in tqdm(range(0, dtm.shape[1])):
        topic = dtm.getcol(i).nonzero()[0]
        lens.append(len(topic))
        if len(topic) == 0:
            zeros += 1
    print("Topic-Doc distributions.")
    print("Minimum: " + str(min(lens)))
    print("Maximum: " + str(max(lens)))
    print("Average: " + str(np.mean(lens)))
    print("Entropy: " + str(entropy(lens, base=len(lens))))
    print("Zeros: " + str(zeros))

    sb.set_theme(style="whitegrid")
    ax = sb.boxplot(x=lens)
    plt.show()

    lens = []
    zeros = 0
    for i in tqdm(range(0, dtm.shape[0])):
        topic = dtm.getrow(i).nonzero()[0]
        lens.append(len(topic))
        if len(topic) == 0:
            zeros += 1
    print("Doc-Topic distributions.")
    print("Minimum: " + str(min(lens)))
    print("Maximum: " + str(max(lens)))
    print("Average: " + str(np.mean(lens)))
    print("Entropy: " + str(entropy(lens, base=len(lens))))
    print("Zeros: " + str(zeros))


def run_lda(path: str, cv_matrix, words, corpus):
    # fitting the lda model and saving it
    lda = fit_lda(cv_matrix, words)
    save_lda(lda, path)

    # saving document topics to file
    print("creating document topics file")
    create_document_topics(corpus, lda)

    return lda


def load_dict_file(path, separator=','):
    csv_reader = pd.read_csv(path, header=None, encoding='unicode_escape', sep=separator)
    test = dict(csv_reader.values.tolist())
    return test


if __name__ == '__main__':
    # Loading data and preprocessing
    model_path = 'model_test'
    cv = sp.load_npz("../Generated Files/count_vec_matrix.npz")
    words = load_dict_file("../Generated Files/word2vec.csv")
    mini_corpus = load_dict_file("../Generated Files/doc2word.csv", separator='-')
    mini_corpus = [x[1:-1].split(', ') for x in mini_corpus.values()]
    mini_corpus = [[int(y) for y in x] for x in mini_corpus]
    run_lda('/model/', cv, words, mini_corpus)
