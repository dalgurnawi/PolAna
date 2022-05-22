# Tristan word2vec embedding for word similarity from colab notebook

import os
import logging
import os
import multiprocessing
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')


class Sentences(object):
    def __init__(self):
        self.sentence_count = 0
        self.epoch = 0

    def __iter__(self):
        print(f"Epoch {self.epoch}")
        self.epoch += 1

        files = os.listdir(".")
        files = [file for file in files if file.endswith(".txt")]
        for fname in files:
            with open(fname, encoding="unicode_escape") as f_input:
                corpus = f_input.read()
            raw_sentences = sent_tokenize(corpus)
            for sentence in raw_sentences:
                if len(sentence) > 0:
                    self.sentence_count += 1
                    yield simple_preprocess(sentence)


sentences = Sentences()

model = Word2Vec(
    sg=1,
    size=3000,
    window=20,
    min_count=3,
    workers=multiprocessing.cpu_count())

model.build_vocab(sentences)

model.train(sentences=sentences, total_examples=model.corpus_count, epochs=5)

print("Done.")

print(model.wv.most_similar("russian", topn=1000))
print(len(model.wv.most_similar("russian", topn=10)))
print(len(model.wv.most_similar("russian", topn=100)))
print(len(model.wv.most_similar("russian", topn=1000)))
print(model.wv.vocab.items())