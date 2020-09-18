from typing import Tuple, Any, List
import networkx as net
import nltk
import math

from preprocessing import load_document_file


def document_graph(path_to_file: str) -> net.graph:
    documents = list(load_document_file(path_to_file).values())

    # Sentence graph
    document_graph = net.Graph()

    # iterating over sentences and adding them as nodes
    # comparing all sentences to each other.
    for document in documents:
        document_graph.add_node(document)

        for second_document in documents:
            words_in_sent2_value = len(words_in_sent2)
            overlap = list(set(words_in_sent) & set(words_in_sent2))
            overlap_value = len(overlap)
            similarity_co_occurrence = overlap_value / (
                        math.log10(words_in_sent_value) + math.log10(words_in_sent2_value))
            sentence_graph.add_edge(sentence, second_sentence, weight=similarity_co_occurrence)
    return document_graph


if __name__ == '__main__':
    graph = document_graph("documents.json")
