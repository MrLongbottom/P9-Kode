import re
import json
import numpy as np

from multiprocessing import Pool
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
word_maximum_doc_percent = 0.20
doc_minimum_length = 20


def main():
    print('Beginning Word2Vec Procedure.')

    # load documents file
    print('Beginning to load documents from "' + load_filename + '".')
    load_file(load_filename)
    print('Finished loading ' + str(len(documents)) + ' documents from "' + load_filename + '".')

    # transform documents into a dict containing unique word counts
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)
    print(X.shape)
    words = cv.get_feature_names()
    print('Found ' + str(len(words)) + " unique words.")

    tf = np.sum(X, axis=0)
    for i in range(0, len(tf)):
        if tf[0][i] < word_minimum_count:
            np.delete(X, i, axis=0)

    """
    # save word2vec file
    print('Beginning to save Word2Vec in "' + save_filename + '".')
    save_file(save_filename, word_count_dict)
    print('Finished Saved Word2Vec in "' + save_filename + '".')

    print('Finished Word2Vec Procedure.')
    return
    """

def preprocess_words(word_count_dict, document_word_dict, word_doc_count_dict):
    # Full TF-IDF

    #Old versions
    word_count_dict = {key: word_count_dict[key] for key in word_count_dict if
                       word_count_dict[key] > 20}
    docs_num = len(document_word_dict)
    word_count_dict = {key: word_count_dict[key] for key in word_count_dict if
                       word_doc_count_dict[key] / docs_num < word_maximum_doc_percent}
    return word_count_dict


def word_to_vec():
    word_count_dict = {}
    document_word_dict = {}
    word_doc_count_dict = {}
    with Pool(pool_size) as p:
        word_counts = p.map(word2vec_doc_load, documents)
        for wcs in tqdm(word_counts):
            if wcs is None:
                continue
            words = []
            for wc in wcs[0]:
                words.append(wc[0])
                word_count_dict[wc[0]] = word_count_dict.get(wc[0], 0) + wc[1]
                word_doc_count_dict[wc[0]] = word_doc_count_dict.get(wc[0], 0) + 1
            document_word_dict[wcs[1]] = words

    return word_count_dict, document_word_dict, word_doc_count_dict


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


def save_file(filename, dict):
    with open(filename, "w") as file:
        id_counter = 0
        for w2v in dict.items():
            file.write(str(id_counter) + ", " + str(w2v[0]) + ", " + str(w2v[1]) + '\n')
            id_counter += 1


def word2vec_doc_load(doc):
    # combine headline and body into test
    text = doc['headline'] + " " + doc['body']
    text = text.lower()
    # basic word regex filter
    words = re.findall(r"[a-zæøå]+", text)
    # removing documents that contain too few words
    if len(words) < doc_minimum_length:
        return None
    unique_words = []
    count = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
            count.append(1)
        else:
            count[unique_words.index(word)] += 1
    return [x for x in zip(unique_words, count)], doc['id']

if __name__ == "__main__":
    main()
