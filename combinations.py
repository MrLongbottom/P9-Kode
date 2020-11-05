import language_model
import query_handling
import utility
import numpy as np

if __name__ == '__main__':
    queries = utility.load_vector_file("Generated Files/queries.csv")
    query_key = list(queries.keys())[0]
    query_content = list(queries.values())[0].split(' ')
    res1, p_vec1 = query_handling.lda_evaluate_query(query_key, query_content, tell=True)
    res2, p_vec2 = language_model.lm_evaluate_query(query_key, query_content, tell=True)
    p_vec3 = [p_vec1[i] * p_vec2[i] for i in range(len(p_vec1))]
    print(str(res1) + " " + str(res2))
    print(f"lda: {p_vec1[query_key]}, lm: {p_vec2[query_key]}, combo: {p_vec3[query_key]}")
