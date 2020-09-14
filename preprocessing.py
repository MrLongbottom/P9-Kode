import re
import json
import numpy as np

from multiprocessing import Pool
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

# formatted as: ID, header, body
documents = {}
corpus = []
# name of file to load words from
load_filename = "documents.json"
# name of file to save down
save_filename = "word2vec.txt"
pool_size = 4
word_minimum_count = 20
word_maximum_doc_percent = 0.25
doc_minimum_length = 20


def main(save):
    print('Beginning Word2Vec Procedure.')

    # load documents file
    print('Beginning to load documents from "' + load_filename + '".')
    load_file(load_filename)
    print('Loaded ' + str(len(documents)) + ' documents.')

    # transform documents into a dict containing unique word counts
    cv = CountVectorizer(max_df=word_maximum_doc_percent, min_df=word_minimum_count)
    tf = TfidfTransformer()
    X = cv.fit_transform(corpus)
    X2 = tf.fit_transform(X)
    words = cv.get_feature_names()
    print('Found ' + str(len(words)) + " unique words.")

    if save:
        save_file(save_filename, words)
        save_file('doc2vec.txt', documents)
    return X


def load_file(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        index = 0
        for json_obj in json_file:
            data = json.loads(json_obj)
            index += 1
            text = data['headline'] + " " + data['body']
            text = text.lower()
            text = re.findall(r"[a-zæøå]+", text)
            if len(text) < doc_minimum_length:
                continue
            text = ' '.join(text)
            corpus.append(text)
            documents[index] = data['id']


def save_file(filename, words):
    with open(filename, "w") as file:
        id_counter = 0
        for w in words:
            file.write(str(id_counter) + ", " + str(w) + '\n')
            id_counter += 1


if __name__ == "__main__":
    main(True)
