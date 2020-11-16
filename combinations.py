import utility
from query_handling import evaluate_query
from query_evaluation import lm_evaluate_word_doc, lda_evaluate_word_doc, lm_lda_combo_evaluate_word_doc


def lm_lda_combo():
    # Load queries
    queries = utility.load_vector_file("Generated Files/queries.csv")

    for query_key, query_content in queries.items():
        # Evaluate the queries
        res, p_vec = evaluate_query(lm_lda_combo_evaluate_word_doc, query_key, query_content.split(' '), tell=False)
        # List the results
        print(f"original query index: {query_key} original query: {query_content}")
        print(f"LDA*LM index:{res}")


if __name__ == '__main__':
    lm_lda_combo()
