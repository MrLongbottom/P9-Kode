import json
import math

from nltk.corpus import stopwords
# nltk.download('stopwords')
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer


def load_file(filename):
    documents = []
    with open(filename, "r", encoding="utf-8") as json_file:
        for json_obj in json_file:
            data = json.loads(json_obj)
            documents.append(data)
    return documents


def lda_model(k: int, data):
    number_topics = k
    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(data)
    return lda


def count_vectorizer(data):
    cnt_vectorizer = CountVectorizer(stop_words=stopwords.words('danish'))
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
    data = load_file("../2018_data.json")[:1000]

    # num of topics
    num_of_topics = math.floor(math.sqrt(len(data)))

    # vectorize data
    vectorizer, data = count_vectorizer([x['body'] for x in data])

    # train model
    model = lda_model(num_of_topics, data.toarray())

    # print topics
    print_topics(model, vectorizer, 10)
