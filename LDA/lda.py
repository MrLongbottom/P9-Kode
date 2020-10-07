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
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.stats import entropy
from tqdm import tqdm
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


def run_lda(path: str, cv_matrix, words, corpus, save_path):
    # fitting the lda model and saving it
    lda = fit_lda(cv_matrix, words)
    save_lda(lda, path)

    # saving document topics to file
    print("creating document topics file")
    td_matrix = create_document_topics(corpus, lda, save_path + "topic_doc_matrix.npz")
    # saving topic words to file
    print("creating topic words file")
    tw_matrix = save_topic_word_matrix(lda, save_path + "topic_word_matrix.npz")

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