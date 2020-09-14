import re
import json

from multiprocessing import Pool
from tqdm import tqdm

# formatted as: ID, header, body
documents = []
# name of file to load words from
load_filename = "documents.json"
# name of file to save down
save_filename = "word2vec.txt"
pool_size = 4
word_minimum_count = 20
word_maximum_percent = 25


def main():
    print('Beginning Word2Vec Procedure.')

    # load documents file
    print('Beginning to load documents from "' + load_filename + '".')
    load_file(load_filename)
    print('Finished Loaded Word2Vec from "' + load_filename + '".')

    # transform documents into a dict containing unique word counts
    word_count_dict = word_to_vec()
    print('Found ' + str(len(word_count_dict)) + " unique words.")

    # preprocess to remove unhelpful words
    word_count_dict = {key: word_count_dict[key] for key in word_count_dict if word_count_dict[key] < 20}
    print('Found ' + str(len(word_count_dict)) + " unique words.")

    # save word2vec file
    print('Beginning to save Word2Vec in "' + save_filename + '".')
    save_file(save_filename, word_count_dict)
    print('Finished Saved Word2Vec in "' + save_filename + '".')

    print('Finished Word2Vec Procedure.')
    return


def word_to_vec():
    word_count_dict = {}
    with Pool(pool_size) as p:
        word_counts = p.map(word2vec_doc_load, documents)
        for wcs in tqdm(word_counts):
            for wc in wcs:
                word_count_dict[wc[0]] = word_count_dict.get(wc[0], 0) + wc[1]
    return word_count_dict


def load_file(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        for json_obj in json_file:
            data = json.loads(json_obj)
            documents.append(data)


def save_file(filename, dict):
    with open(filename, "w") as file:
        id_counter = 0
        for w2v in dict.items():
            file.write(str(id_counter) + ", " + str(w2v[0]) + ", " + str(w2v[1]) + '\n')
            id_counter += 1


def word2vec_doc_load(doc):
    string = doc['headline'] + " " + doc['body']
    string.lower()
    # basic word regex filter
    words = re.findall(r"[a-zæøå]+", string)
    unique_words = []
    count = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
            count.append(1)
        else:
            count[unique_words.index(word)] += 1
    return [x for x in zip(unique_words, count)]


if __name__ == "__main__":
    main()
