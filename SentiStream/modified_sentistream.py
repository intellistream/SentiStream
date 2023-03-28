import sys
import logging
import numpy as np
import shutil
import pandas as pd

from pyflink.datastream import CheckpointingMode, StreamExecutionEnvironment
from pyflink.datastream.execution_mode import RuntimeExecutionMode

import config
from modified_classifier import classifier
from modified_batch_inferrence import batch_inference
from modified_evaluation import generate_new_label, merged_stream
from modified_supervised_model import supervised_model
from modified_PLStream import unsupervised_stream
from utils import load_data

# logger
logger = logging.getLogger('PLStream')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('plstream.log', mode='w')
formatter = logging.Formatter('PLStream:%(thread)d %(lineno)d: %(levelname)s: %(asctime)s %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p', )
fh.setFormatter(formatter)
logger.addHandler(fh)

# supress warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO, format="%(message)s")

    parallelism = 1

    pseudo_data_folder = './senti_output'
    test_data_file = 'exp_test.csv'
    train_data_file = 'exp_train.csv'

    ## -------------------INITIAL TRAINING OF SUPERVISED MODEL------------------- ##
    new_df = pd.read_csv('train.csv', names=['label', 'review'])
    new_df['label'] -= 1

    df = new_df[0:100].reset_index()

    true_label = df.label
    yelp_review = df.review

    data_stream = []

    for i in range(len(yelp_review)):
        data_stream.append((int(true_label[i]), yelp_review[i]))

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.BATCH)
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

    print('Starting SentiStream...')
    print('===============================')

    ds = env.from_collection(collection=data_stream)

    if supervised_model(ds, parallelism, len(data_stream), 0, 0, init=True):
        env.execute()

    for k in range(1, 2):

        df = new_df[k * 100: (k+1) * 100].reset_index()

        ## -------------------GENERATE PSEUDO-LABEL FROM BOTH LEARNING METHODS------------------- ##
        true_label = df.label
        yelp_review = df.review

        data_stream = []

        for i in range(len(yelp_review)):
            data_stream.append((i, int(true_label[i]), yelp_review[i]))

        # env.set_parallelism(1)
        env.set_runtime_mode(RuntimeExecutionMode.STREAMING)

        ds = env.from_collection(collection=data_stream)

        print("unsupervised stream,classifier and evaluation")

        # data stream functions
        # (idx, conf, pred, label, {neg, pos, true})
        ds1 = unsupervised_stream(ds)
        ds2 = classifier(ds)  # (idx, conf, pred, label)

        # ds2.print()
        ds = merged_stream(ds1, ds2)
        ds = generate_new_label(ds)
        env.execute()

        ## -------------------SUPERVISED MODEL INFERENCE------------------- ##

        # data sets prep
        pseudo_data_size, test_df = load_data(
            pseudo_data_folder, test_data_file)

        test_df = df

        true_label = test_df.label
        yelp_review = test_df.review

        data_stream = []

        for i in range(len(yelp_review)):
            data_stream.append((int(true_label[i]), yelp_review[i]))

        print("batch_inference")

        # env.set_parallelism(1)

        env.set_runtime_mode(RuntimeExecutionMode.BATCH)

        ds = env.from_collection(collection=data_stream)

        accuracy = batch_inference(ds, len(test_df))
        print(accuracy)

        ## -------------------SUPERVISED MODEL TRAIN-------------------##
        print("supervised_model_train")

        config.PSEUDO_DATA_COLLECTION_THRESHOLD = 0
        config.ACCURACY_THRESHOLD = 0.9

        # train model on pseudo data with supervised mode
        pseudo_data_size, train_df = load_data(
            pseudo_data_folder, train_data_file)

        train_df = df

        # env.set_parallelism(1)
        env.set_runtime_mode(RuntimeExecutionMode.BATCH)

        true_label = df.label
        yelp_review = df.review

        data_stream = []

        for i in range(len(yelp_review)):
            data_stream.append((int(true_label[i]), yelp_review[i]))

        ds = env.from_collection(collection=data_stream)

        if supervised_model(ds, parallelism, len(data_stream), pseudo_data_size, 0.4):
            env.execute()

        shutil.rmtree('senti_output', ignore_errors=False, onerror=None)
