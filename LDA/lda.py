import json
import math

from sklearn.decomposition import LatentDirichletAllocation as LDA


def load_file(filename):
    documents = []
    with open(filename, "r", encoding="utf-8") as json_file:
        for json_obj in json_file:
            data = json.loads(json_obj)
            documents.append(data)
    return documents


def lda_model(k: int, data):
    # Tweak the two parameters below
    number_topics = k
    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(data)
    return lda


def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


if __name__ == '__main__':
    data = load_file("../2018_data.json")[:1000]
    # model = lda_model(math.floor(math.sqrt(len(pre_data))), pre_data)
    # print(model)
