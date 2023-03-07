import re
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from torch.utils.data import DataLoader, Dataset
from pyflink.common.typeinfo import Types
from pyflink.common.serialization import Encoder
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import MapFunction
from pyflink.datastream import CheckpointingMode
from pyflink.datastream.connectors import StreamingFileSink
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SentimentDataset(Dataset):
    def __init__(self, vector, label):
        self.vector = vector
        self.label = label

    def __len__(self):
        return len(self.vector)
    
    def __getitem__(self, idx):
        return self.vector[idx], self.label[idx]
    

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim_2, output_dim=1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

class SupervisedOSA(MapFunction):

    def __init__(self, collector_size=2000):
        # collection
        self.true_label = []
        self.cleaned_text = []
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
             "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
             'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
             'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
             'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
             'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
             'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o',
             're', 've', 'y', 'ma', 'st', 'nd', 'rd', 'th', "you'll", 'dr', 'mr', 'mrs']
        self.tokenizer = RegexpTokenizer(r'[a-z]+')
        self.collector_size = collector_size

        # w2v model
        self.w2v = None
        self.vec_dim = 50

        # ann 
        self.classifier = None
        self.batch_size = 128

        # results
        self.predictions = []

    # tweet preprocessing
    def text_to_word_list(self, text):
        text = text.lower()
        text = re.sub("@\w+ ", "", text)
        tokens = self.tokenizer.tokenize(text)
        clean_word_list = [word for word in tokens if word not in self.stop_words]
        while '' in clean_word_list:
            clean_word_list.remove('')
        self.cleaned_text.append(clean_word_list)

        if len(self.cleaned_text) >= self.collector_size:
            if self.classifier == None:
                ans = self.train_classifier(self.cleaned_text)
                return ans
            else:
                embeddings = self.get_sent_embeddings(self.w2v, self.cleaned_text)
                classify_result = self.eval(embeddings, self.classifier)
                return ('acc', classify_result)
        else:
            return ('collecting', '1')

    def map(self, tweet):
        self.true_label.append(int(tweet[1]))
        return self.text_to_word_list(tweet[0])
    
    def get_sent_embeddings(self, model, sents):
        embeddings = []

        for sent in sents:
            wv_vec = np.zeros(self.vec_dim)
            count = 0

            for word in sent:
                if word in model.wv:
                    count += 1
                    wv_vec += model.wv[word]
            
            if count > 0:
                wv_vec /= count

            embeddings.append(wv_vec)
        return embeddings

    
    def train_w2v(self, tokens):
        self.w2v = Word2Vec(vector_size=self.vec_dim, window=5, min_count=3, workers=4)
        self.w2v.build_vocab(tokens)
        self.w2v.train(tokens, total_examples=self.w2v.corpus_count, epochs=10)

    def train_ann(self, model, train_loader, val_loader, criterion, optimizer):
        EPOCHS = 500
        best_val_loss = 10
        for _ in range(EPOCHS):
            # train
            model.train()

            for vecs, labels in train_loader:
                # forward
                outputs = model(vecs)
                loss = criterion(outputs, labels)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # validation
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for vecs, labels in val_loader:
                    outputs = model(vecs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                self.classifier = model

        self.classifier.eval()
    
    def train_classifier(self, tokens):
        self.train_w2v(tokens)
        sent_embeddings = self.get_sent_embeddings(self.w2v, tokens)

        x = torch.tensor(np.array(sent_embeddings), dtype=torch.float32)
        y = torch.tensor(np.array(self.true_label), dtype=torch.float32).unsqueeze(1)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

        train_data = SentimentDataset(x_train, y_train)
        val_data = SentimentDataset(x_val, y_val)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        model = Classifier(input_dim=self.vec_dim, hidden_dim=16, hidden_dim_2=32)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5)

        self.train_ann(model, train_loader, val_loader, criterion, optimizer)

        self.cleaned_text = []
        self.true_label = []

        return ('model', '0')


    def eval(self, tweets, model):
        self.predictions = self.predict(tweets, model)

        ans = accuracy_score(self.true_label, self.predictions)
        self.predictions = []

        return str(ans)

    def predict(self, tweets, model):

        steps = len(tweets) // self.batch_size + (0 if len(tweets) % self.batch_size == 0 else 1)
        pred = []
        with torch.no_grad():
            for i in range(steps):
                pred += torch.round(model(torch.tensor(tweets[i*self.batch_size : (i+1)*self.batch_size], dtype=torch.float32))).tolist()

        return pred

if __name__ == '__main__':
    parallelism = 2
    
    # the labels of dataset are only used for accuracy computation, since PLStream is unsupervised
    df = pd.read_csv('train.csv', names=['label', 'review'])
    
    # 80,000 data for quick testing
    df = df.iloc[:2000,:]

    df['label'] -= 1

    true_label = list(df.label)
    yelp_review = list(df.review)

    data_stream = []
    
    for i in range(len(yelp_review)):
        data_stream.append((yelp_review[i], int(true_label[i])))

    print('Coming Stream is ready...')
    print('===============================')

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    ds = env.from_collection(collection=data_stream)
    ds = ds.map(SupervisedOSA(int(len(yelp_review) * 0.05))).set_parallelism(parallelism) \
        .filter(lambda x: x[0] != 'collecting') \
        .key_by(lambda x: x[0], key_type=Types.STRING()) \
        .filter(lambda x: x[0] != 'model') \
        .map(lambda x: str(x[1]), output_type=Types.STRING()).set_parallelism(1) \
        .add_sink(StreamingFileSink
                  .for_row_format('./output', Encoder.simple_string_encoder())
                  .build())

    # ds.print()
    env.execute()
