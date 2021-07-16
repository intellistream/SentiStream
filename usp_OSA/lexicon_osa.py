from time import time
from re import sub
import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import pandas as pd #著名数据处理包
import nltk 
from nltk import word_tokenize #分词函数
from nltk.corpus import stopwords #停止词表，如a,the等不重要的词
from nltk.corpus import sentiwordnet as swn #得到单词情感得分
import string #本文用它导入标点符号，如!"#$%& 
# logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pyflink.datastream.functions import RuntimeContext, MapFunction
#from pyflink.common.serialization import SimpleStringEncoder
from pyflink.common.typeinfo import TypeInformation, Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
#from pyflink.datastream.connectors import StreamingFileSink

class Lexicon_OSA(MapFunction):

    def __init__(self):
        self.stop = stopwords.words("english")
        self.noun = ['NN','NNP','NNPS','NNS','UH']
        self.verb = ['VB','VBD','VBG','VBN','VBP','VBZ']
        self.adj = ['JJ','JJR','JJS']
        self.adv = ['RB','RBR','RBS','RP','WRB']
        self.result = []

    def open(self, runtime_context: RuntimeContext):

        # initial test data for evalaution
        self.test_data = pd.read_csv('osatest.csv')
        self.clean_test_data = self.test_data.tweet.apply(lambda x: self.text_to_word_list(x))
    
    def map(self,text):
        tweet = text.lower() #change to lower case
        tweet = re.sub("@\w+ ","", tweet) #removes all usernames in text
        tweet = re.sub("\'s", " ", tweet) # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
        tweet = re.sub("\'t", " not ", tweet)
        tweet = re.sub(" whats ", " what is ", tweet, flags=re.IGNORECASE)
        tweet = re.sub("\'ve", " have ", tweet)
        tweet = re.sub('[^a-zA-Z]', ' ', tweet)  #remove numbers
        tweet = re.sub(r'\s+', ' ', tweet)      
        tweet = re.sub(r'[^\w\s]','',tweet)     #remove commas
        tweet = re.sub('(?<=[0-9])\,(?=[0-9])', "", tweet) # remove comma between numbers, i.e. 15,000 -> 15000
        clean_tweet = re.sub("[!~#$+%*:()'?-]", ' ', tweet)   # remove characters stated below
        return clean_tweet
    
    def part_of_speech(self,text):
        word_form = []
        word_tag = nltk.pos_tag([i for i in word_tokenize(str(text)) if i not in self.stop])
        for key in word_tag:
            if key[1] in self.noun:
                word_form.append(f'{key[0]}.n.01')
            elif key[1] in self.verb:
                word_form.append(f'{key[0]}.v.01')
            elif key[1] in self.adj:
                word_form.append(f'{key[0]}.a.01')
            elif key[1] in self.adv:
                word_form.append(f'{key[0]}.r.01')
            else:
                word_form.append('')
        return word_form

    def eval(self,sentence):
        pos_score,neg_score = 0,0
        for word in sentence:
            try:
                pos_score += swn.senti_synset(word).pos_score()
                neg_score += swn.senti_synset(word).neg_score()
            except:
                pos_score += 0
        if pos_score >= neg_score:
            self.result.append(4)
        elif pos_score < neg_score:
            self.result.append(0)
        if len(self.result) >=40000:
            ans = accuracy_score(self.result,[0]*20000+[4]*20000)
            return str(ans)
        else:
            return 'next'

if __name__ == '__main__':

    from pyflink.datastream.connectors import StreamingFileSink
    from pyflink.common.serialization import Encoder
    coming_tweets = pd.read_csv('flinktestdata.csv')
    coming_tweets = list(coming_tweets.tweet)[180000:220000]
    print('Coming tweets is ready...')
    print('===============================')

    #final = []
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    
    ds = env.from_collection(collection = coming_tweets)

    ds.map(lambda x: Lexicon_OSA().map(x))\
      .map(lambda x: Lexicon_OSA().part_of_speech(x))\
      .map(lambda x: Lexicon_OSA().eval(x))\
      .add_sink(StreamingFileSink
      .for_row_format('./output', Encoder.simple_string_encoder())
      .build())
    env.execute("osa_job")
