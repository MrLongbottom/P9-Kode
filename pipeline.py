from gensim.corpora import Dictionary

from LDA.lda import run_lda
from preprocessing import preprocess

if __name__ == '__main__':
    # Setup pipeline for the project
    # Preprocess
    cv_matrix, words, corpus = preprocess("documents.json")

    # Run LDA

    run_lda(path="LDA/model/docu_model_sqrt_div2",
            cv_matrix=cv_matrix,
            words=words,
            corpus=corpus,
            save_path="Generated Files/",
            dictionary=Dictionary(corpus))

    # Make Graph
    # graph()
