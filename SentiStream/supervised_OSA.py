import nltk
nltk.download('stopwords')
import random
import copy
from numpy import dot
from numpy.linalg import norm
import re
import numpy as np
from gensim.models import Word2Vec
from gensim import models
import redis
import pickle
import logging
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.common.typeinfo import TypeInformation, Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream import CheckpointingMode
import nltk

'''
Supevised mode:
BERT for sentence embedding,
MLP for classification
'''

import torch
from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')

class BertTextNet(nn.Module):
    def __init__(self, code_length, model_name='bert-base-uncased'): 
        super(BertTextNet, self).__init__()
 
        modelConfig = BertConfig.from_pretrained(model_name)
        # self.textExtractor = BertModel.from_pretrained(
        #     'bert-base-chinese-pytorch_model.bin', config=modelConfig)
        self.textExtractor = BertModel.from_pretrained(model_name)
        embedding_dim = self.textExtractor.config.hidden_size
 
        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()
 
    def forward(self, tokens, segments, input_masks):
        output=self.textExtractor(tokens, token_type_ids=segments,
                                 		attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]  
        #output[0](batch size, sequence length, model hidden dimension)
 
        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features

class MLP(nn.Module):
    def __init__(self, input_dim, out_dim=2, hidden_dim=512):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.flatten = nn.Flatten()
        # self.in_layer = nn.Linear(input_dim, hidden_dim)
        # self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        # self.out_layer = nn.Linear(hidden_dim, out_dim)

        self.stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax(dim=1) # needed to modify to softmax
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.stack(x)
        return x


class supervised_OSA(MapFunction):
    def __init__(self, phase='eval'):
        print('supervised OSA')
        # self.model = svm.SVC()
        self.phase = phase
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        self.clf = MLP(8)
        self.bert = None
        self.bert = BertTextNet(8)
        # self.bert = MLP(20)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.clf.parameters())

        # for test:
        self.true_labels = []
        self.stop_words = stopwords.words('english')
        self.sno = nltk.stem.SnowballStemmer('english')
        self.thousand_text = []
        self.predictions = []
  
    def SentenceEmbedding(self, sentences):
        print('-----sentence embedding-----')
        # texts = ['I am a student at university.']
        tokens, segments, input_masks = [], [], []
        for sentence in sentences:
            tokenized_text = self.tokenizer.tokenize(sentence) #get tokens
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)#index list
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            input_masks.append([1] * len(indexed_tokens))

        max_len = max([len(single) for single in tokens]) #maximum sentence length

        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding

        tokens_tensor = torch.tensor(tokens) # sentence embedding
        segments_tensors = torch.tensor(segments)
        input_masks_tensors = torch.tensor(input_masks)

        # features = None
        features = self.bert(tokens_tensor, segments_tensors, input_masks_tensors)

        return features

    def predict(self, tweet):
        sentence_embedding_vec = self.SentenceEmbedding(tweet)
        print('vec {} for {}'.format(sentence_embedding_vec. self.phase))

        if self.phase == 'train':
            return
            self.train_clf_model(sentence_embedding_vec)
        elif self.phase == 'eval':
            return
            self.predictions.append(self.clf(sentence_embedding_vec))
            # print(self.predictions)

            ans = accuracy_score(self.true_labels, self.predictions)

        # return self.clf(sentence_embedding_vec)
        # return self.clf(x)

    def train_clf_model(self, embedding_vec):
        
        y_pred = self.clf(embedding_vec)
        loss = self.criterion(y_pred, self.true_labels[-1])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def eval(self, tweets, labels):
        # no eval at present
        pred = []
        for tweet in tweets:
            pred.append(self.predict(tweet))

        ans = accuracy_score(labels, self.predictions)
        return str(ans)

    def open(self, runtime_context: RuntimeContext):
        self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)

        # data for initial model
        self.initial_model = models.Word2Vec.load('/content/drive/My Drive/ogb_10.model')
        self.vocabulary = self.initial_model.wv.key_to_index

    def map(self, tweet):
        # print('map')
        # the entry of the whole function
        self.true_labels.append(int(tweet[1]))
        return self.text_to_word_list(tweet[0])

    # tweet preprocessing
    def text_to_word_list(self, tweet):
        # print(tweet)
        tweet = tweet.lower()  # change to lower case
        tweet = re.sub("@\w+ ", "", tweet)  # removes all usernames in text
        tweet = re.sub("\'s", " ", tweet)
        tweet = re.sub("\'t", " ", tweet)
        tweet = re.sub("[!~#$+%*:()'?-]", ' ', tweet)  # remove characters stated below
        tweet = re.sub(r'\s+', ' ', tweet)
        tweet = re.sub(r'[^\w\s]', '', tweet)  # remove commas
        tweet = re.sub('[^a-zA-Z]', ' ', tweet)  # remove numbers

        # print(tweet)
        print('go to prediction')
        self.predict(tweet)
            
    def print_test(self, x):
        print(x)
        print('==================')
        return x

# for Supervised

if __name__ == '__main__':
    from pyflink.datastream.connectors import StreamingFileSink
    from pyflink.common.serialization import Encoder
    from pyflink.datastream.connectors import FileSink, OutputFileConfig
    import sys
    import pandas as pd
    from time import time

    dataset = 'yelp' # use yelp data to test

    f = pd.read_csv('./train.csv')  #, encoding='ISO-8859-1'
    true_label = list(f.polarity)[:40000]
    yelp_review = list(f.tweet)[:40000]
    data_stream = []
    for i in range(len(yelp_review)):  # len(true_label) 40000):
        data_stream.append((yelp_review[i], int(true_label[i])))
    # print(len(data_stream))

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    process_time = time()
    # env.get_checkpoint_config().set_checkpointing_mode(checkpointing_mode=)
    ds = env.from_collection(collection=data_stream)  #, output_type=Types.STRING()
    
    ds = ds.map(supervised_OSA()).set_parallelism(1)\
        .filter(lambda x: x[0] != 'collecting')\
        .key_by(lambda x: x[0], key_type=Types.STRING())
        # .map(supervised_OSA().map).set_parallelism(2)
        # .map(supervised_OSA().print_test).set_parallelism(2)
        # .filter(lambda x: x[0] != 'model')\
        # .map(for_output(), output_type=Types.STRING()).set_parallelism(1)\
        # .add_sink(StreamingFileSink   #.set_parallelism(2)
        #           .for_row_format('./output', Encoder.simple_string_encoder())
        #           .build())
    ds.print()
    env.execute("osa_job")
    process_time_total = (time() - process_time) / 60
    print(f"SentiStream solution has finished the job, total time cost:{process_time_total} minutes")
