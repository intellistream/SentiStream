import codecs as cs
import numbers
import _JST
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pyflink.datastream.functions import RuntimeContext, MapFunction
# from pyflink.common.serialization import SimpleStringEncoder
from pyflink.common.typeinfo import TypeInformation, Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment


class JST_OSA(MapFunction):

    def __init__(self, topics=1, sentilab=3, iteration=100, K=50,
                 beta=0.01, gamma=0.01, random_state=123456789,
                 refresh=50):
        self.topics = topics
        self.sentilab = sentilab
        self.iter = iteration
        self.alpha = (K + .0) / (topics + .0)
        self.beta = beta
        self.gamma = gamma
        self.random_state = random_state
        self.refresh = refresh
        if self.alpha <= 0 or beta <= 0 or gamma <= 0:
            raise ValueError("alpha,beta and gamma must be greater than zero")

        rng = self.check_random_state(random_state)
        self._rands = rng.rand(1024 ** 2 // 8)
        self.true_label = []
        self.vocab_size = 0
        self.word2id = {}
        self.id2word = {}
        self.vocab = set()
        self.docs = []
        self.doc_num = 0
        self.doc_size = 0
        self.stop_words = stopwords.words('english')
        for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's']:
            self.stop_words.append(w)
        self.accuracy_lda = []
        self.counter = 1

    def open(self, runtime_context: RuntimeContext):
        prior_information = 2
        if prior_information == 1:
            self.read_model_prior(r'./constraint/mpqa.constraint')
        elif prior_information == 2:
            self.read_model_prior(r'/home/huilin/Documents/Learning/experimentOSA/JST_model-on-MR-master/constraint/paradigm_words.constraint')
        elif prior_information == 3:
            self.read_model_prior(r'./constraint/full_subjectivity_lexicon.constraint')
        elif prior_information == 4:
            self.read_model_prior(r'./constraint/filter_lexicon.constraint')
        elif prior_information == 0:
            self.read_model_prior(r'./constraint/empty_prior')
        self.thread_index = runtime_context.get_index_of_this_subtask()


    def reset_model(self, topics=1, sentilab=3, iteration=100, K=50,
                 beta=0.01, gamma=0.01, random_state=123456789,
                 refresh=50):
        self.topics = topics
        self.sentilab = sentilab
        self.iter = iteration
        self.alpha = (K + .0) / (topics + .0)
        self.beta = beta
        self.gamma = gamma
        self.random_state = random_state
        self.refresh = refresh
        if self.alpha <= 0 or beta <= 0 or gamma <= 0:
            raise ValueError("alpha,beta and gamma must be greater than zero")
        rng = self.check_random_state(random_state)
        self._rands = rng.rand(1024 ** 2 // 8)
        self.vocab_size = 0
        self.word2id = {}
        self.id2word = {}
        self.vocab = set()
        self.docs = []
        self.doc_num = 0
        self.doc_size = 0
        self.true_label = []


    def read_model_prior(self, prior_path):
        """Joint Sentiment-Topic Model using collapsed Gibbs sampling.
        Sentiment lexicon or other methods.
        The format of the prior imformation are as follow :
        [word]  [neu prior prob.] [pos prior prob.] [neg prior prob.]
        ...
        """
        self.prior = {}
        model_prior = cs.open(prior_path, 'r')
        for word_prior in model_prior.readlines():
            word_prior = word_prior.strip().split()
            index = 1
            maxm = -1.0
            for i in range(1, len(word_prior)):
                word_prior[i] = float(word_prior[i])
                if word_prior[i] > maxm:
                    maxm = word_prior[i]
                    index = i
            self.prior[word_prior[0]] = word_prior[1:]
            self.prior[word_prior[0]].append(index - 1)

    def check_random_state(self, seed):
        if seed is None:
            # i.e., use existing RandomState
            return np.random.mtrand._rand
        if isinstance(seed, (numbers.Integral, np.integer)):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError("{} cannot be used as a random seed.".format(seed))

    def clean_text(self, tweet_with_label):

        tweet = tweet_with_label.lower()  # change to lower case
        tweet = re.sub("@\w+ ", "", tweet)  # removes all usernames in text
        tweet = re.sub("\'s", " ",
                       tweet)  # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable
        tweet = re.sub("\'t", " not ", tweet)
        tweet = re.sub(" whats ", " what is ", tweet, flags=re.IGNORECASE)
        tweet = re.sub("\'ve", " have ", tweet)
        tweet = re.sub('[^a-zA-Z]', ' ', tweet)  # remove numbers
        tweet = re.sub(r'\s+', ' ', tweet)
        tweet = re.sub(r'[^\w\s]', '', tweet)  # remove commas
        tweet = re.sub('(?<=[0-9])\,(?=[0-9])', "", tweet)  # remove comma between numbers, i.e. 15,000 -> 15000
        tweet = re.sub("[!~#$+%*:()'?-]", ' ', tweet)  # remove characters stated below
        clean_word_list = tweet.split(' ')
        clean_word_list = [w for w in clean_word_list if w not in self.stop_words]
        while '' in clean_word_list:
            clean_word_list.remove('')
        for item in clean_word_list:
            self.vocab.add(item)
            self.doc_size += 1

        # corpus collector
        self.docs.append(clean_word_list)

        if len(self.docs) >= 5000:
            self.analyze_corpus()
            self.init_model_parameters()
            self.estimate()
            self.cal_pi_ld()
            ans = self.eval()
            self.reset_model(topics=1, sentilab=3, iteration=100, K=50,
                 beta=0.01, gamma=0.01, random_state=123456789,
                 refresh=50)
            return ans
        else:
            return 'collecting'

    def analyze_corpus(self):

        self.vocab_size = len(self.vocab)
        index = 0
        for item in self.vocab:
            self.word2id[item] = index
            self.id2word[index] = item
            index += 1

    def add_prior(self):
        # beta add prior information
        for word in self.prior:
            if word in self.vocab:
                for l in range(self.sentilab):
                    self.add_lw[l][self.word2id[word]] *= self.prior[word][l]

    def init_model_parameters(self):
        # model counts
        self.nd = np.zeros((self.doc_num,), dtype=np.int32)
        self.ndl = np.zeros((self.doc_num, self.sentilab), dtype=np.int32)
        self.ndlz = np.zeros((self.doc_num, self.sentilab, self.topics), dtype=np.int32)
        self.nlzw = np.zeros((self.sentilab, self.topics, self.vocab_size), dtype=np.int32)
        self.nlz = np.zeros((self.sentilab, self.topics), dtype=np.int32)

        # model parameters
        self.pi_dl = np.zeros((self.doc_num, self.sentilab), dtype=np.float)
        self.theta_dlz = np.zeros((self.doc_num, self.sentilab, self.topics), dtype=np.float)
        self.phi_lzw = np.zeros((self.sentilab, self.topics, self.vocab_size), dtype=np.float)

        # init hyperparameters with prior imformation
        self.alpha_lz = np.full((self.sentilab, self.topics), fill_value=self.alpha)
        self.alphasum_l = np.full((self.sentilab,), fill_value=self.alpha * self.topics)

        if (self.beta <= 0):
            self.beta = 0.01

        self.beta_lzw = np.full((self.sentilab, self.topics, self.vocab_size), fill_value=self.beta)
        self.betasum_lz = np.zeros((self.sentilab, self.topics), dtype=np.float)

        # #word prior
        self.add_lw = np.ones((self.sentilab, self.vocab_size), dtype=np.float)

        self.add_prior()
        for l in range(self.sentilab):
            for z in range(self.topics):
                for r in range(self.vocab_size):
                    self.beta_lzw[l][z][r] *= self.add_lw[l][r]
                    self.betasum_lz[l][z] += self.beta_lzw[l][z][r]

        if self.gamma <= 0:
            self.gamma = 1.0

        self.gamma_dl = np.full((self.doc_num, self.sentilab), fill_value=0.0)
        self.gammasum_d = np.full((self.doc_num,), fill_value=.0)
        for d in range(self.doc_num):
            # self.gamma_dl[d][1] = 1.8
            self.gamma_dl[d][1] = self.gamma
            self.gamma_dl[d][2] = self.gamma
        for d in range(self.doc_num):
            for l in range(self.sentilab):
                self.gammasum_d[d] += self.gamma_dl[d][l]

    def estimate(self):
        random_state = self.check_random_state(self.random_state)
        rands = self._rands.copy()
        self.init_estimate()
        ll = ll_pre = 0.0
        for it in range(self.iter):
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                ll += self.loglikelihood()
                if ll / (it / self.refresh + 1) - 10 <= ll_pre and it > 0:
                    break
                ll_pre = ll / (it / self.refresh + 1)
            self._sampling(rands)

    def init_estimate(self):

        self.ZS = []
        self.LS = []
        self.WS = []
        self.DS = []
        self.IS = []

        cnt = 1
        prior_word_cnt = 0
        for m, doc in enumerate(self.docs):
            for t, word in enumerate(doc):
                cnt += 1
                if word in self.prior:
                    senti = self.prior[word][-1]
                    self.IS.append(int(1))
                    prior_word_cnt += 1
                else:
                    # senti = int(np.random.uniform(0,self.sentilab))
                    senti = (cnt) % self.sentilab
                    self.IS.append(int(0))
                # topi = int(np.random.uniform(0,self.topics))
                topi = (cnt) % self.topics
                self.DS.append(int(m))
                self.WS.append(int(self.word2id[word]))
                self.LS.append(int(senti))
                self.ZS.append(int(topi))

                self.nd[m] += 1
                self.ndl[m][senti] += 1
                self.ndlz[m][senti][topi] += 1
                self.nlzw[senti][topi][self.word2id[word]] += 1
                self.nlz[senti][topi] += 1

        self.DS = np.array(self.DS, dtype=np.int32)
        self.WS = np.array(self.WS, dtype=np.int32)
        self.LS = np.array(self.LS, dtype=np.int32)
        self.ZS = np.array(self.ZS, dtype=np.int32)
        self.IS = np.array(self.IS, dtype=np.int32)

    def map(self, tweet):

        self.true_label.append(int(tweet[1]))
        self.doc_num += 1
        clean_list = self.clean_text(tweet[0])
        return clean_list

    def loglikelihood(self):
        """Calculate complete log likelihood, log p(w,z,l)
        Formula used is log p(w,z,l) = log p(w|z,l) + log p(z|l,d) + log p(l|d)
        """
        nd, ndl, ndlz, nlzw, nlz = self.nd, self.ndl, self.ndlz, self.nlzw, self.nlz
        return _JST._loglikelihood(nd, ndl, ndlz, nlzw, nlz, self.alpha, self.beta, self.gamma)

    def _sampling(self, rands):

        _JST._sample_topics(self.nd, self.ndl, self.ndlz, self.nlzw, self.nlz,
                            self.alpha_lz, self.alphasum_l, self.beta_lzw, self.betasum_lz,
                            self.gamma_dl, self.gammasum_d, self.DS, self.WS, self.LS, self.ZS, self.IS,
                            rands)

    def cal_pi_ld(self):
        for d in range(self.doc_num):
            for l in range(self.sentilab):
                self.pi_dl[d][l] = (self.ndl[d][l] + self.gamma_dl[d][l]) / (self.nd[d] + self.gammasum_d[d])

    def cal_theta_dlz(self):
        for d in range(self.doc_num):
            for l in range(self.sentilab):
                for z in range(self.topics):
                    self.theta_dlz[d][l][z] = (self.ndlz[d][l][z] + self.alpha_lz[l][z]) \
                                              / (self.ndl[d][l] + self.alphasum_l[l])

    def cal_phi_lzw(self):
        for l in range(self.sentilab):
            for z in range(self.topics):
                for w in range(self.vocab_size):
                    self.phi_lzw[l][z][w] = (self.nlzw[l][z][w] + self.beta_lzw[l][z][w]) \
                                            / (self.nlz[l][z] + self.betasum_lz[l][z])

    def eval(self):

        polarity = []
        for d in range(self.doc_num):
            if self.pi_dl[d][1] > self.pi_dl[d][2]:
                polarity.append(2)
            else:
                polarity.append(1)
        acc = accuracy_score(self.true_label, polarity)
        self.accuracy_lda.append(acc)
        self.accuracy_lda.append(acc)
        if self.counter > 10:
            result = str(self.accuracy_lda)
            return result
        else:
            return '==========' + str(acc) + '==========='

    def plot_result(self, y, num):
        import matplotlib.pyplot as plt
        x = [0]
        for i in range(1, num):
            x.append(f'b{i}')
            x.append(f'b{i}')
        x.append(f'b{num}')
        plt.plot(x, y)
        plt.title("Lexicon-based Online Sentiment Analysis Performance")
        plt.xlabel('batch_size = 2000')
        plt.ylabel('accuracy')
        plt.grid()
        plt.savefig(f'./lexicon_{self.thread_index}.png')
        return True



if __name__ == '__main__':
    from pyflink.datastream.connectors import StreamingFileSink
    from pyflink.common.serialization import Encoder

    l = 40000
    f = pd.read_csv('./train.csv')
    true_label = list(f.polarity)[:l]
    yelp_review = list(f.tweet)[:l]
    yelp_stream = []
    for i in range(l):
        yelp_stream.append((yelp_review[i], true_label[i]))

    print('Coming tweets is ready...')
    print('===============================')

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    ds = env.from_collection(collection=yelp_stream)
    ds.map(JST_OSA(), output_type=Types.STRING())\
        .add_sink(StreamingFileSink
        .for_row_format('./output', Encoder.simple_string_encoder())
        .build())
    env.execute("osa_job")
