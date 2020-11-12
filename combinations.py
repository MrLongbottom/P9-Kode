import utility
import scipy.sparse as sp
from query_handling import evaluate_query
from query_evaluation import lm_evaluate_word_doc, lda_evaluate_word_doc, lm_lda_combo_evaluate_word_doc


def investigate_results():
    # Load queries
    queries = utility.load_vector_file("Generated Files/queries.csv")

    for query_key, query_content in queries.items():
        # Evaluate the queries using the two models
        res1, p_vec1 = evaluate_query(lda_evaluate_word_doc, query_key, query_content, tell=False)
        res2, p_vec2 = evaluate_query(lm_evaluate_word_doc, query_key, query_content, tell=False)

        # Combine the two models
        p_vec3 = {k: p_vec1[k] * p_vec2[k] for k in range(len(p_vec1))}
        res3 = list(dict(sorted(p_vec3.items(), key=lambda x: x[1], reverse=True)).keys()).index(query_key)

        # List the results
        print(f"LDA index: {res1}, LM index:{res2} LDA*LM index:{res3}")
        print(
            f"LDA index value: {p_vec1[query_key]}, LM index value: {p_vec2[query_key]}, LDA*LM index value: {p_vec3[query_key]}")


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
