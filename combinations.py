import utility
import scipy.sparse as sp
from query_handling import evaluate_query
from query_evaluation import lm_evaluate_word_doc, lda_evaluate_word_doc, lm_lda_combo_evaluate_word_doc


def lm_lda_combo():
    # Load queries and matrices
    queries = utility.load_vector_file("Generated Files/queries.csv")
    dt_matrix = sp.load_npz("Generated Files/topic_doc_matrix.npz")
    tw_matrix = sp.load_npz("Generated Files/topic_word_matrix.npz")

    for query_key, query_content in queries.items():
        # Evaluate the queries
        res, p_vec = evaluate_query(lm_lda_combo_evaluate_word_doc,
                                    query_key,
                                    query_content.split(' '),
                                    (dt_matrix, tw_matrix),
                                    tell=False)
        # List the results
        print(f"original query index: {query_key} original query: {query_content}")
        print(f"LDA*LM index:{res}")


if __name__ == '__main__':
    lm_lda_combo()
