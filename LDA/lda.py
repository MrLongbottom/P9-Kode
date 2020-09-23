import math
from multiprocessing import Pool
from typing import Dict, List

import scipy.sparse as sp
from scipy.stats import entropy
import numpy as np
from gensim import matutils
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from gensim.test.utils import datapath
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import seaborn as sb
from matplotlib import pyplot as plt

from preprocessing import load_word_files


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
                        num_topics=math.floor(math.sqrt(data.shape[1])))


def save_lda(lda: LdaModel, path: str):
    # Save model to disk.
    temp_file = datapath(path)
    lda.save(temp_file)


def load_lda(path: str):
    return LdaModel.load(path)


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
    query = lda.get_document_topics(corpus, minimum_probability=0.025)
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


def run_lda(path: str, cv_matrix, words, mini_corpus):
    # fitting the lda model and saving it
    lda = fit_lda(cv_matrix, words)
    save_lda(lda_model, model_path)

    # saving document topics to file
    print("creating document topics file")
    create_document_topics(mini_corpus)

    return lda


if __name__ == '__main__':
    # Loading data and preprocessing
    model_path = 'model_test'
    cv = sp.load_npz("../Generated Files/count_vec_matrix.npz")
    words, mini_corpus = load_word_files(["../Generated Files/word2vec.csv", "../Generated Files/doc2word.csv"])

    # Fitting the model and saving it
    lda_model = fit_lda(cv, words)
    save_lda(lda_model, model_path)

    # Loading the model
    # lda = load_lda(model_path)

    # Saving Docs
    doc_topic_matrix = create_document_topics(mini_corpus)
    print("test")
