import math
from gensim.corpora import Dictionary

from lda import run_lda
from preprocessing import preprocess

if __name__ == '__main__':
    # Setup pipeline for the project
    # Preprocess
    cv_matrix, vocab, documents = preprocess("data/2017_data.json")
    corpus = [vocab.doc2bow(doc) for doc in documents]

    # Run LDA
    K = math.floor(math.sqrt(cv_matrix.shape[0]) / 2)
    params = (K, None, 0.001)

    run_lda("LDA/model/document_model" + str(params),
            documents=documents,
            corpus=corpus,
            vocab=vocab,
            save_path="Generated Files/",
            param_combination=params)
