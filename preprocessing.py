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
word_save_filename = "word2vec.txt"
doc_save_filename = "doc2vec.txt"
pool_size = 4
word_minimum_count = 20
word_maximum_doc_percent = 0.25
doc_minimum_length = 20


def main(save):
    print('Beginning Word2Vec Procedure.')

    # load documents file
    load_file(load_filename)

    # transform documents into a matrix containing counts for each word in each document
    # also cut off words that are used too often or too little (max/min document frequency)
    cv = CountVectorizer(max_df=word_maximum_doc_percent, min_df=word_minimum_count)
    X = cv.fit_transform(corpus)
    # calculate term frequency - inverse document frequency
    # (might not be needed)
    tf = TfidfTransformer()
    X2 = tf.fit_transform(X)
    words = cv.get_feature_names()
    print('Found ' + str(len(words)) + " unique words.")

    if save:
        print('Saving word and document lookup files.')
        save_file(word_save_filename, words)
        save_file(doc_save_filename, documents.values())
    return X


def load_file(filename):
    print('Beginning to load documents from "' + load_filename + '".')
    with open(filename, "r", encoding="utf-8") as json_file:
        index = 0
        for json_obj in json_file:
            data = json.loads(json_obj)
            text = data['headline'] + " " + data['body']
            text = text.lower()
            text = re.findall(r"[a-zæøå]+", text)
            # Remove documents containing too few words.
            if len(text) < doc_minimum_length:
                continue
            text = ' '.join(text)
            corpus.append(text)
            documents[index] = data['id']
            index += 1
    print('Loaded ' + str(len(documents)) + 'valid documents.')


def save_file(filename, words):
    with open(filename, "w") as file:
        id_counter = 0
        for w in words:
            file.write(str(id_counter) + ", " + str(w) + '\n')
            id_counter += 1


if __name__ == "__main__":
    main(True)
