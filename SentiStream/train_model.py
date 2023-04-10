# import redis
# from ann_model import Model
from han_model import Model
from gensim.models import Word2Vec
from utils import pre_process, default_model_pretrain, train_word2vec, generate_vector_mean, clean_text_w2v

# pre_processs - label, string


class InitialModelTrain:
    def __init__(self, data):
        # self.w2v_model = default_model_pretrain("PLS_c10.model")
        self.w2v_model = Word2Vec(vector_size=20, window=5, min_count=5, workers=12)
        # self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

        self.labels = []
        self.texts = []

        self.train_models(data)

    def train_classifier(self):
        # clf = Model(mean_vectors, self.labels,
        #             self.w2v_model.vector_size, True)
        clf = Model(self.texts, self.labels, self.w2v_model.wv.key_to_index, [self.w2v_model.wv[key] for key in self.w2v_model.wv.index_to_key], True)
        clf.fit_and_save('ssl-clf.pth')

        # TODO CONTINOUS TRAIN

    def train_w2v(self):
        self.w2v_model.build_vocab(self.texts)
        self.w2v_model.train(self.texts,
                    total_examples=self.w2v_model.corpus_count,
                    epochs=30)
        self.w2v_model.save('ssl-w2v.model')

    def train_models(self, data):
        for d in data:
            # self.sentences.append(d[1])
            # label, text = pre_process(d)
            # self.labels.append(label)
            # self.texts.append(text)
            self.labels.append(d[0])
            self.texts.append(clean_text_w2v(d[1]))

        # train_word2vec(self.w2v_model, self.texts, 'ssl-w2v.model')
        self.train_w2v()

        # mean_vectors = [generate_vector_mean(self.w2v_model, text) for text in self.texts]

        # self.train_classifier(mean_vectors)

        self.train_classifier()

        # try:
        #     self.redis.set('word_vector_update', int(True))
        #     self.redis.set('classifier_update', int(True))
        # except ConnectionError:
        #     raise ConnectionError('Failed to open redis')