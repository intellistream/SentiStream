# import redis
from ann_model import Model
from utils import pre_process, default_model_pretrain, train_word2vec, generate_vector_mean

# pre_processs - label, string


class InitialModelTrain:
    def __init__(self, data):
        self.w2v_model = default_model_pretrain("PLS_c10.model")
        # self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

        self.labels = []
        self.texts = []

        self.train_models(data)

    def train_classifier(self, mean_vectors):
        clf = Model(mean_vectors, self.labels,
                    self.w2v_model.vector_size, True)
        clf.fit_and_save('model.pth')

        # TODO CONTINOUS TRAIN

    def train_models(self, data):
        for d in data:
            label, text = pre_process(d)
            self.labels.append(label)
            self.texts.append(text)

        train_word2vec(self.w2v_model, self.texts, 'w2v.model')

        mean_vectors = [generate_vector_mean(self.w2v_model, text) for text in self.texts]

        self.train_classifier(mean_vectors)

        # try:
        #     self.redis.set('word_vector_update', int(True))
        #     self.redis.set('classifier_update', int(True))
        # except ConnectionError:
        #     raise ConnectionError('Failed to open redis')