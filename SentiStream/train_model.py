import redis
from ann_model import Model
from utils import pre_process, default_model_pretrain, train_word2vec, generate_vector_mean

# pre_processs - label, string


class InitialModelTrain:
    def __init__(self, tweets):
        self.w2v_model = default_model_pretrain("PLS_c10.model")
        self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

        self.labels = []
        self.sentences = []

        self.train_models(tweets)

    def train_classifier(self, mean_vectors):
        clf = Model(mean_vectors, self.labels,
                    self.w2v_model.vector_size, True)
        clf.fit_and_save('model.pth')

        # TODO CONTINOUS TRAIN

    def train_models(self, tweets):
        for tweet in tweets:
            label, sentence = pre_process(tweet)
            self.labels.append(label)
            self.sentences.append(sentence)

        self.train_wordvector_model()

        mean_vectors = []
        for sentence in self.sentences:
            mean_vectors.append(generate_vector_mean(self.w2v_model, sentence))

        self.train_classifier(mean_vectors)

        try:
            self.redis.set('word_vector_update', int(True))
            self.redis.set('classifier_update', int(True))
        except ConnectionError:
            raise ConnectionError('Failed to open redis')

    def train_wordvector_model(self, func=train_word2vec):
        func(self.w2v_model, self.sentences, 'w2v.model')
