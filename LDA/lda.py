import itertools
import json
import math

from boto.cloudfront.object import Object
from nltk.corpus import stopwords
# nltk.download('stopwords')
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


def get_file_generator(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        for json_obj in json_file:
            yield json.loads(json_obj)['body']


def get_length_of_file(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        return sum(1 for line in json_file)


def itertive_model_train(num_samples: int, k_topics: int, data):
    lda = LDA(n_components=k_topics, n_jobs=-1)
    for sample in tqdm(range(num_samples)):
        slice = next(data).toarray()
        lda.partial_fit(slice)
    return lda


def count_vectorizer(data):
    cnt_vectorizer = CountVectorizer(stop_words=stopwords.words('danish'))
    count_data = cnt_vectorizer.fit_transform(data)
    return cnt_vectorizer, iter(count_data)


def print_topics(model, cnt_vectorizer, n_top_words):
    words = cnt_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


if __name__ == '__main__':

    file_name = "../documents.json"
    # Load data
    data = get_file_generator(file_name)

    # num of topics
    num_of_topics = math.floor(math.sqrt(get_length_of_file(file_name)))
    print(f"number of topics: {num_of_topics}")
    # vectorize data
    vectorizer, count_data = count_vectorizer(data)
    print("Vectorizer finished")
    # train model
    # model = lda_model(num_of_topics, data.toarray())
    model = itertive_model_train(get_length_of_file(file_name), num_of_topics, count_data)

    # print topics
    print_topics(model, vectorizer, 10)
