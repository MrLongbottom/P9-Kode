from multiprocessing import Pool
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as net
import numpy as np
import scipy.sparse as sp
import seaborn as sb
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from scipy.stats import entropy
from tqdm import tqdm


def similarity_between_documents(d1: int, d2: int):
    """
    Computes similarity between two documents by summing the minimum of their topic distribution.
    :param d1: document 1
    :param d2: document 2
    :return: a similarity value between 0 and 1
    """
    return sum(
        [min(topic_doc_matrix.getrow(d1).toarray()[0][number], topic_doc_matrix.getrow(d2).toarray()[0][number]) for
         number in set(topic_doc_matrix.getrow(0).nonzero()[1]) & set(topic_doc_matrix.getrow(1).nonzero()[1])])


def create_document_topics(corpus: List[str]) -> sp.dok_matrix:
    """
    Creates a topic_doc_matrix which describes the amount of topics in each document
    :param corpus: list of document strings
    :return: a topic_document matrix
    """
    document_topics = []
    with Pool(8) as p:
        document_topics.append(p.map(get_document_topics_from_model, corpus))
    matrix = save_topic_doc_matrix(document_topics[0])
    return matrix


def get_document_topics_from_model(text: str) -> Dict[int, float]:
    """
    A method used concurrently in create_document_topics
    :param text: a document string
    :return: a dict with the topics in the given document based on the lda model
    """
    tokenized_text = [text.split(' ')]
    dictionary = Dictionary(tokenized_text)
    corpus = [dictionary.doc2bow(t) for t in tokenized_text]
    query = lda_model.get_document_topics(corpus, minimum_probability=0.025)
    return dict([x for x in query][0])


def save_topic_doc_matrix(document_topics: List[Dict[int, float]]) -> sp.dok_matrix:
    """
    Saves the document topics (list of dicts) in a matrix
    :param document_topics: list of dicts
    :return: a matrix (scipy)
    """
    matrix = sp.dok_matrix((len(document_topics), lda_model.num_topics))
    for index, dictionary in tqdm(enumerate(document_topics)):
        for dict_key, dict_value in dictionary.items():
            matrix[index, dict_key] = dict_value
    sp.save_npz("Generated Files/topic_doc_matrix", sp.csc_matrix(matrix))
    return matrix


def document_similarity_matrix(matrix) -> net.graph:
    """
    Todo still not optimized
    """
    length = matrix.shape[0]
    doc_sim_matrix = sp.dok_matrix((length, length))
    for index in tqdm(range(length)):
        for second_index in range(index):
            sim = similarity_between_documents(index, second_index)
            doc_sim_matrix[index, second_index] = sim
    sp.save_npz("Generated Files/doc_sim_matrix", doc_sim_matrix)
    return doc_sim_matrix


# def parallel(index, document):
#     for second_index in range(index):
#         sim = similarity_between_documents(document_topics[index], document_topics[second_index])
#         doc_sim_matrix[index, second_index] = sim
#
#
# def parallel_doc_sim_matrix(document_topics):
#     with Pool(8) as p:
#         p.starmap(parallel, [x for x in enumerate(document_topics)])


def evaluate_topics(dtm):
    lens = []
    zeros = 0
    for i in tqdm(range(0, dtm.shape[1])):
        topic = dtm.getcol(i).nonzero()[0]
        lens.append(len(topic))
        if len(topic) == 0:
            zeros += 1
    print("Minimum: " + str(min(lens)))
    print("Maximum: " + str(max(lens)))
    print("Average: " + str(np.mean(lens)))
    print("Entropy: " + str(entropy(lens, base=len(lens))))
    print("Zeros: " + str(zeros))
    sb.set_theme(style="whitegrid")
    ax = sb.boxplot(x=lens)
    plt.show()


if __name__ == '__main__':
    # Loading stuff and initialisation
    topic_doc_matrix = sp.load_npz("Generated Files/topic_doc_matrix.npz")
    lda_model = LdaModel.load("LDA/model/docu_model")

    # documents = preprocess('documents.json')
    # create_document_topics(documents[2])
