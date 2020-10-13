import pyLDAvis.gensim
from gensim.corpora import Dictionary

from LDA.lda import load_lda, load_corpus

if __name__ == '__main__':
    lda_model = load_lda("LDA/model/docu_model_sqrt_div2")
    texts = load_corpus("Generated Files/corpus")
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
