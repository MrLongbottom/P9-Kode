import ast
import json
from multiprocessing import Pool
from typing import Dict, List

import networkx as net
import numpy as np
import scipy.sparse as sp
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from tqdm import tqdm


def similarity_between_documents(d1: Dict[int, float], d2: Dict[int, float]):
    return sum([min(d1[number], d2[number]) for number in set(d1) & set(d2)])


def construct_transistion_probability_matrix(graph: net.Graph) -> np.ndarray:
    """
    This function constructs a transistion probability matrix
    based on the adjacency matrix of the graph
    :param graph: the sentence graph
    :return: np.array
    """
    adj_matrix = net.adj_matrix(graph).todense()
    row_normalized_matrix = adj_matrix / adj_matrix.sum(axis=0)
    return np.array(row_normalized_matrix, dtype=np.float64)


def get_document_topics_from_model(text: str):
    dictionary = Dictionary([text])
    corpus = [dictionary.doc2bow(t) for t in [text]]

    query = lda_model.get_document_topics(corpus)

    return dict([x for x in query][0])


def load_document_topics(path: str):
    with open(path, 'r') as file:
        document_topics = json.load(file)
    return [ast.literal_eval(x) for x in document_topics]


def create_document_topics(corpus):
    document_topics = []
    with Pool(8) as p:
        document_topics.append(p.map(get_document_topics_from_model, corpus))
    with open('document_topics.json', 'w') as file:
        json.dump(document_topics, file)
    return document_topics


def save_topic_doc_matrix(document_topics: List[Dict[int, float]]):
    matrix = sp.dok_matrix((len(document_topics), lda_model.num_topics))
    for index, dictionary in enumerate(document_topics):
        for dict_key, dict_value in dictionary.items():
            matrix[index, dict_key] = dict_value

    return sp.save_npz("Generated Files/topic_doc_matrix", sp.csc_matrix(matrix))


def document_graph(document_topics=None):
    if document_topics is None:
        document_topics = load_document_topics("document_topics.json")

    for index in tqdm(range(len(document_topics))):
        doc_sim_vector = np.zeros(len(document_topics))
        for second_index in range(index):
            sim = similarity_between_documents(document_topics[index], document_topics[second_index])
            doc_sim_vector[second_index] = sim
        results['data'][index] = doc_sim_vector


def parallel(index):
    doc_sim_vector = np.zeros(len(document_topics))
    for second_index in range(index):
        sim = similarity_between_documents(document_topics[index], document_topics[second_index])
        doc_sim_vector[second_index] = sim
    results['data'][index] = doc_sim_vector


def parallel_doc_sim_matrix(document_topics):
    with Pool(8) as p:
        p.map(parallel, range(len(document_topics)))


if __name__ == '__main__':
    # Loading stuff
    document_topics = load_document_topics("document_topics.json")
    lda_model = LdaModel.load("LDA/model/docu_model")

    import h5py

    hdf5_store = h5py.File("./doc_sim_matrix.hdf5", "a")
    res = hdf5_store.create_dataset("data", (len(document_topics), len(document_topics)), compression="gzip")

    results = h5py.File('doc_sim_matrix.hdf5', 'a')

    document_graph(document_topics)
