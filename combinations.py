import scipy.sparse as sp

import utility
from query_handling import lm_evaluate_word_doc, lda_evaluate_word_doc, evaluate_query


def lm_lda_combo():
    queries = utility.load_vector_file("Generated Files/queries.csv")
    query_key = list(queries.keys())[0]
    query_content = list(queries.values())[0].split(' ')
    dt = sp.load_npz("Generated Files/topic_doc_matrix.npz")
    tw = sp.load_npz("Generated Files/topic_word_matrix.npz")
    res1, p_vec1 = evaluate_query(lda_evaluate_word_doc, query_key, query_content, tell=True)
    res2, p_vec2 = evaluate_query(lm_evaluate_word_doc, query_key, query_content, tell=True)
    p_vec3 = {k: p_vec1[k] * p_vec2[k] for k in range(len(p_vec1))}
    res3 = list(dict(sorted(p_vec3.items(), key=lambda x: x[1], reverse=True)).keys()).index(query_key)
    print(f"LDA index: {res1}, LM index:{res2} LDA*LM index:{res3}")
    print(
        f"LDA index value: {p_vec1[query_key]}, LM index value: {p_vec2[query_key]}, LDA*LM index value: {p_vec3[query_key]}")


if __name__ == '__main__':
    lm_lda_combo()
