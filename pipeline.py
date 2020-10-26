import math
from gensim.corpora import Dictionary

from lda import run_lda
from preprocessing import preprocess

if __name__ == '__main__':
    # Setup pipeline for the project
    # Preprocess
    cv_matrix, words, corpus = preprocess("documents.json")

    # Run LDA
    K = math.floor(math.sqrt(cv_matrix.shape[0]) / 2)
    run_lda(path="LDA/model/document_model",
            cv_matrix=cv_matrix,
            words=words,
            corpus=corpus,
            save_path="Generated Files/",
            dictionary=Dictionary(corpus),
            K=K)
