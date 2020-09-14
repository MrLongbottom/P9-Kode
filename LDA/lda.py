import itertools
import json
import math

from boto.cloudfront.object import Object
from nltk.corpus import stopwords
# nltk.download('stopwords')
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


def load_file(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        for json_obj in json_file:
            yield json.loads(json_obj)


def get_length_of_file(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        return sum(1 for line in json_file)


def train_lda_model(lda: LDA, data):
    return lda.partial_fit(data)


def itertive_model_train(n_slices: int, k_topics: int, data):
    lda = LDA(n_components=k_topics, n_jobs=-1)
    data_slices = itertools.islice(data, n_slices)
    for slice in tqdm(data_slices):
        lda = train_lda_model(lda, slice)
    return lda


def count_vectorizer(data):
    cnt_vectorizer = CountVectorizer(stop_words=stopwords.words('danish'))
    count_data = []
    data_slices = itertools.islice(data, 1000)
    for slice in tqdm(data_slices):
        count_data = cnt_vectorizer.fit_transform(data)
    return cnt_vectorizer, count_data


def print_topics(model, cnt_vectorizer, n_top_words):
    words = cnt_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


if __name__ == '__main__':
    # Load data
    data = load_file("../2018_data.json")

    # num of topics
    num_of_topics = math.floor(math.sqrt(get_length_of_file("../2018_data.json")))
    print(num_of_topics)
    # vectorize data
    vectorizer, data = count_vectorizer([x['body'] for x in data])

    # train model
    # model = lda_model(num_of_topics, data.toarray())
    model = itertive_model_train(100, num_of_topics, data.toarray())

    # print topics
    print_topics(model, vectorizer, 10)
