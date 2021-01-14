from models.lda import run_lda
from preprocessing import preprocess

if __name__ == '__main__':
    # Setup pipeline for the project
    # Preprocess
    cv_matrix, vocab, documents = preprocess("data/2017_data.json")
    corpus = [vocab.doc2bow(doc) for doc in documents]

    # Run LDA
    K = 30
    params = (K, 0.1, 0.1)

    run_lda("models/final_model",
            documents=documents,
            corpus=corpus,
            vocab=vocab,
            save_path="generated_files/",
            param_combination=params)
