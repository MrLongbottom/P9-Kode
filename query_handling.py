import random
from typing import Dict

import utility
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
import scipy.sparse as sp


def generate_queries(count_matrix, words: Dict[int, str], count: int, min_length: int = 1, max_length: int = 4):
    """
    Generates queries for random documents based on tfidf values
    :param count_matrix: CountVectorization matrix
    :param words: words dictionary
    :param count: number of queries wanted
    :param min_length: min words per query (exact length is random)
    :param max_length: max words per query (exact length is random)
    :return: dictionary mapping document ids to queries
    """
    tfidf = TfidfTransformer()
    tfidf_matrix = tfidf.fit_transform(count_matrix)
    queries = {}
    documents_count = tfidf_matrix.shape[0]
    for i in tqdm(range(count)):
        doc_id = random.randrange(0, documents_count)
        query_length = random.randrange(min_length, max_length+1)
        query = []
        doc_vec = tfidf_matrix.getrow(doc_id)
        word_ids = doc_vec.toarray()[0].argsort()[-query_length:][::-1]
        for word_id in word_ids:
            word = words[word_id]
            query.append(word)
        query = ' '.join(query)
        queries[doc_id] = query
    return queries


if __name__ == '__main__':
    cv_matrix = sp.load_npz("Generated Files/count_vec_matrix.npz")
    word2vec = utility.load_vector_file("Generated Files/word2vec.csv")
    queries = generate_queries(cv_matrix, word2vec, 100, min_length=4, max_length=4)
    doc2word = utility.load_vector_file("Generated Files/doc2word.csv")
    print("test")
    fails = []
    for k, v in queries.items():
        for word in v:
            if word not in doc2word[k]:
                print("Not in there!")
                fails.append((k, v, word, doc2word[k]))

    print("done")