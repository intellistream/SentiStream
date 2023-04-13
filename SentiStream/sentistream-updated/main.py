# pylint: disable=import-error

# US WORD_VECTOR_NAME = 'plstream-wv.model'
# SS WORD_VECTOR_NAME = 'ssl-wv.model'
# CLASSIFIER_NAME = 'clf.pth'

# TORCH JIT ?
# TODO: FIX CLASS IMBALANCE IN INITIAL TRAINING - DONE - REWATCH
# TODO: UPDATE HAN PREPROCESSING FROM 08a914a COMMIT


from time import time
import pandas as pd

from gensim.models import Word2Vec, FastText

from pyflink.datastream import CheckpointingMode, StreamExecutionEnvironment
from train.supervised import TrainModel
from unsupervised_models.plstream import PLStream
from inference.classifier import Classifier


PYFLINK = True
FLINK_PARALLELISM = 1

SSL_MODEL = 'HAN' # 'HAN', 'ANN'
WORD_VEC_ALGO =  FastText # Word2Vec, FastText


# ---------------- Initial training of classifier ----------------
# Train word vector {Word2Vec or FastText} on {nrows=1000} data and get word embeddings, then use
# that embedding to train NN sentiment classifier {ANN or HAN}

start = time()

TrainModel(word_vector_algo=WORD_VEC_ALGO, ssl_model=SSL_MODEL, init=True, nrows=1000)


# ---------------- Generate pesudo labels ----------------

df = pd.read_csv('train.csv', names=[
    'label', 'review'], nrows=1000)
df['label'] -= 1

plstream = PLStream(word_vector_algo=WORD_VEC_ALGO)
classifier = Classifier(word_vector_algo=WORD_VEC_ALGO, ssl_model=SSL_MODEL)

if PYFLINK:
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(FLINK_PARALLELISM)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

    data_stream = [(int(label), review) for label, review in df.values]
    ds = env.from_collection(collection=data_stream)

    ds_us = ds.map(plstream.process_data).filter(lambda x: x != 'BATCHING').print()
    ds_ss = ds.map(classifier.classify).filter(lambda x: x != 'BATCHING').print()

    env.execute()

else:
    for idx, row in df.iterrows():
        results = plstream.process_data((row.label, row.review))

        if results != 'BATCHING':
            print(results)

        results = classifier.classify((row.label, row.review))

        if results != 'BATCHING':
            print(results)

print(time() - start)
