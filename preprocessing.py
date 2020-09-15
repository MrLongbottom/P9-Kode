import json
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm


def preprocess(load_filename="documents.json", word_save_filename="word2vec.txt", doc_save_filename="doc2vec.txt",
               word_minimum_count=20, word_maximum_doc_percent=0.25, doc_minimum_length=20, save=True):
    print('Beginning Word2Vec Procedure.')

    # load documents file
    documents = load_file(load_filename)
    # filter documents and create corpus
    documents, corpus = filter_documents(documents, doc_minimum_length)

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
        save_file(doc_save_filename, documents.keys())
    return X


def load_file(filename):
    print('Loading documents from "' + filename + '".')
    documents = {}
    with open(filename, "r", encoding="utf-8") as json_file:
        for json_obj in json_file:
            data = json.loads(json_obj)
            documents[data['id']] = data['headline'] + ' ' + data['body']
    print('Loaded ' + str(len(documents)) + ' documents.')
    return documents


def filter_documents(documents, doc_minimum_length):
    print('Filtering documents.')
    corpus = []
    bad_docs = []
    for doc in tqdm(documents.items()):
        text = doc[1]
        text = text.lower()
        text = re.findall(r"[a-zæøå]+", text)
        # Remove documents containing too few words.
        if len(text) < doc_minimum_length:
            bad_docs.append(doc)
            continue
        text = ' '.join(text)
        documents[doc[0]] = text
        corpus.append(text)
    for doc in bad_docs:
        del documents[doc[0]]
    print('Filtering documents. ' + str(len(documents)) + ' remaining')
    return documents, corpus


def save_file(filename, content):
    print('Saving file "' + filename + '".')
    with open(filename, "w") as file:
        id_counter = 0
        for c in content:
            file.write(str(id_counter) + ", " + str(c) + '\n')
            id_counter += 1
    print('"' + filename + '" has been saved.')


if __name__ == '__main__':
    preprocess()
