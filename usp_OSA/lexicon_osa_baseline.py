# -*- coding: utf-8 -*-
from nltk import word_tokenize  # 分词函数
import nltk
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pyflink.datastream.functions import RuntimeContext, MapFunction
# from pyflink.common.serialization import SimpleStringEncoder
from pyflink.common.typeinfo import TypeInformation, Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

# from pyflink.datastream.connectors import StreamingFileSink

class unsupervised_OSA(MapFunction):

    def __init__(self):
        self.acc_to_plot = []
        self.counter = 1
        self.stop_words = stopwords.words('english')
        for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's']:
            self.stop_words.append(w)
        self.noun = ['NN', 'NNP', 'NNPS', 'NNS', 'UH']
        self.verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.adj = ['JJ', 'JJR', 'JJS']
        self.adv = ['RB', 'RBR', 'RBS', 'RP', 'WRB']
        self.result = []
        self.true_label = []

    def open(self, runtime_context: RuntimeContext):
        test = pd.read_csv('twitterstream.csv', encoding='gbk')
        self.label = list(test.polarity)
        self.thread_index = runtime_context.get_index_of_this_subtask()

    # tweet preprocessing
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

        return self.part_of_speech(tweet)

    def map(self, tweet):

        self.true_label.append(int(tweet[1]))
        clean_list = self.clean_text(tweet[0])
        return clean_list

    def part_of_speech(self, clean_tweet):

        import nltk
        word_form = []
        word_tag = nltk.pos_tag([i for i in word_tokenize(str(clean_tweet)) if i not in self.stop_words])
        for key in word_tag:
            if key[1] in self.noun:
                word_form.append(f'{key[0]}.n.01')
            # elif key[1] in self.verb:
            #     word_form.append(f'{key[0]}.v.01')
            elif key[1] in self.adj:
                word_form.append(f'{key[0]}.a.01')
            elif key[1] in self.adv:
                word_form.append(f'{key[0]}.r.01')

        return self.eval(word_form)

    def eval(self, word_form):

        from nltk.corpus import sentiwordnet as swn  # 得到单词情感得分
        pos_score, neg_score = 0, 0
        for word in word_form:
            try:
                pos_score += swn.senti_synset(word).pos_score()
                neg_score += swn.senti_synset(word).neg_score()
            except:
                pos_score += 0
        if pos_score >= neg_score:
            self.result.append(4)
        elif pos_score < neg_score:
            self.result.append(0)

        if len(self.result) >= 2000:

            ans = accuracy_score(self.result, self.true_label)
            self.result = []
            self.true_label = []
            self.acc_to_plot.append(ans)
            self.acc_to_plot.append(ans)
            self.counter += 1
            if self.counter > 4:
                return str(self.acc_to_plot)
            else:
                return 'batch'
        else:
            return 'next'

    def plot_result(self, y, num):
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

    parallelism = 2

    # format of input data: (tweet,label)
    twitter = open('./twitter_stream.txt', 'r')
    coming_tweets = []
    for line in twitter:
        tweet_with_label = line.replace('\n', '').split('@whl@')
        coming_tweets.append((tweet_with_label[0], tweet_with_label[1]))

    # coming_tweets = pd.read_csv('twitterstream.csv', encoding='gbk')
    # coming_tweets = list(coming_tweets.tweet)
    print('Coming tweets is ready.')
    print('======================')

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(parallelism)
    print(f'Current parallelism is: {parallelism}')
    ds = env.from_collection(collection=coming_tweets)
    ds.map(unsupervised_OSA(), output_type=Types.STRING()) \
      .print()
        # .add_sink(StreamingFileSink
        #           .for_row_format('./output', Encoder.simple_string_encoder())
        #           .build())
    env.execute("osa_job")
    #
    # a = self.plot_result(self.acc_to_plot, 4)
