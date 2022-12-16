import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary


class LDA:
    def __init__(self, tokenized_doc):
        self.dictionary = corpora.Dictionary(tokenized_doc)
        self.dictionary.filter_extremes(no_below=10, no_above=0.05)
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_doc]

    def making_LDA(self, topic_num, chunksize, iterations, passes, eval_every):
        model = LdaModel(corpus=self.corpus, id2word=self.dictionary,
                         chunksize=chunksize, alpha="auto", eta="auto",
                         iterations=iterations, num_topics=topic_num,
                         passes=passes, eval_every=eval_every)
        return model

    def cp_score(self, model):
        cm = CoherenceModel(model, corpus=self.corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        print("Coherence", coherence)
        print('\nPerplexity', model.log_perplexity(self.corpus))

    def tunning_pass(self, max_pass, step):
        coherences = []
        perplexities = []
        passes = []

        for i in range(max_pass):
            ntopics = 200

            if i % step == 0:
                p = i + 1

                lda4 = LdaModel(self.corpus, id2word=self.dictionary, num_topics=ntopics, iterations=400
                                , passes=p)
                print('epoch', p)
                cm = CoherenceModel(model=lda4, corpus=self.corpus, coherence='u_mass')
                coherence = cm.get_coherence()
                print("Coherence", coherence)
                coherences.append(coherence)
                print("Perplexity", lda4.log_perplexity(self.corpus), '\n\n')
                perplexities.append(lda4.log_perplexity(self.corpus))
                passes.append(p)
            else:
                continue

        return coherences, perplexities, passes

    def tunning_topic(self, max_topic, step):

        coherences = []
        perplexities = []
        topic = []
        ldamodel = []

        for i in range(max_topic):
            if i % step == 0:
                if i == 0:
                    continue
                else:
                    p = i
                    lda = self.making_LDA(p, 9000, 400, 30, 10)
                    print('epoch', p)
                    cm = CoherenceModel(model=lda, corpus=self.corpus, coherence='u_mass')
                    coherence = cm.get_coherence()
                    print("Coherence", coherence)
                    coherences.append(coherence)
                    print("Perplexity", lda.log_perplexity(self.corpus), '\n\n')
                    perplexities.append(lda.log_perplexity(self.corpus))
                    topic.append(p)
                    ldamodel.append(lda)

            else:
                continue

        return [i for i in zip(topic, coherences, perplexities, ldamodel)]
