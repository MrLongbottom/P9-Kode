import re, numpy as np, pandas as pd
import scipy.sparse as sp
import gensim.corpora as corpora
from tqdm import tqdm

from lda import load_lda, load_corpus


def format_topics_sentences(ldamodel, corpus, term_doc_freq):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    print("Getting main topics...")
    max_ = len(term_doc_freq)
    with tqdm(total=max_) as pbar:
        for i, row_list in enumerate(ldamodel[term_doc_freq]):
            row = row_list[0] if ldamodel.per_word_topics else row_list
            #print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
            pbar.update()
    print("Finished getting main topics")
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(corpus)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


if __name__ == '__main__':
    lda_model = load_lda("LDA/model/document_model")
    corpus = load_corpus("Generated Files/corpus")
    cv_matrix = sp.load_npz("Generated Files/count_vec_matrix.npz")

    # Create Dictionary
    id2word = corpora.Dictionary(corpus)
    # Create Corpus: Term Document Frequency
    tdf = [id2word.doc2bow(text) for text in corpus]


    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, term_doc_freq=tdf)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.head(10)
