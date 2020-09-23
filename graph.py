from multiprocessing import Pool
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

from preprocessing import preprocess


def similarity_between_documents(d1: Dict[int, float], d2: Dict[int, float]):
    return sum([min(d1[number], d2[number]) for number in set(d1) & set(d2)])


def load_document_topics(path: str):
    with open(path, 'r') as file:
        document_topics = json.load(file)
    return [ast.literal_eval(x) for x in document_topics][0]


def create_document_topics(corpus: List[str]):
    document_topics = []
    with Pool(8) as p:
        document_topics.append(p.map(get_document_topics_from_model, corpus))
    with open('document_topics.json', 'w') as file:
        json.dump([str(document_topics[0])], file)
    return document_topics


def get_document_topics_from_model(text: str):
    tokenized_text = [text.split(' ')]

    dictionary = Dictionary(tokenized_text)
    corpus = [dictionary.doc2bow(t) for t in tokenized_text]

    query = lda_model.get_document_topics(corpus, minimum_probability=0.025)

    return dict([x for x in query][0])


def save_topic_doc_matrix(document_topics: List[Dict[int, float]]):
    matrix = sp.dok_matrix((len(document_topics), lda_model.num_topics))
    for index, dictionary in enumerate(document_topics):
        for dict_key, dict_value in dictionary.items():
            matrix[index, dict_key] = dict_value

    return sp.save_npz("Generated Files/topic_doc_matrix", sp.csc_matrix(matrix))


def document_graph(document_topics=None) -> net.graph:
    if document_topics is None:
        document_topics = load_document_topics("document_topics.json")

    doc_sim_matrix = sp.dok_matrix((len(document_topics), len(document_topics)))
    for index, document in tqdm(enumerate(document_topics)):
        for second_index in range(index):
            sim = similarity_between_documents(document_topics[index], document_topics[second_index])
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


if __name__ == '__main__':
    # Loading stuff and initialisation
    # document_topics = load_document_topics("document_topics.json")
    # doc_sim_matrix = sp.dok_matrix((len(document_topics), len(document_topics)))
    lda_model = LdaModel.load("LDA/model/docu_model")

    # documents = preprocess('documents.json')
    # create_document_topics(documents[2])
    documet_topics = load_document_topics('document_topics.json')
    print("hello")
    # parallel_doc_sim_matrix(document_topics)
