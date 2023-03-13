import sys
import logging
import numpy as np
import pandas as pd

from pyflink.datastream import CheckpointingMode, StreamExecutionEnvironment
from pyflink.datastream.execution_mode import RuntimeExecutionMode

import config
from dummy_classifier import dummy_classifier
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

    ## -------------------INITIAL TRAINING OF SUPERVISED MODEL------------------- ##
    df = pd.read_csv('exp_train.csv', names=['label', 'review'])
    df['label'] -= 1

    supervised_model(parallelism, df, 0, 0, init=True)

    ## -------------------GENERATE PSEUDO-LABEL FROM BOTH LEARNING METHODS------------------- ##
    true_label = df.label
    yelp_review = df.review

    data_stream = []

    for i in range(len(yelp_review)):
        data_stream.append((i, int(true_label[i]), yelp_review[i]))

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    ds = env.from_collection(collection=data_stream)

    print("unsupervised stream,classifier and evaluation")
    print('Coming Stream is ready...')
    print('===============================')

    # data stream functions
    ds1 = unsupervised_stream(ds)
    ds2 = dummy_classifier(ds)
    ds = merged_stream(ds1, ds2)
    ds = generate_new_label(ds)
    env.execute()

    print("Finished running datastream")

    ## -------------------SUPERVISED MODEL INFERENCE------------------- ##
    pseudo_data_folder = './senti_output'
    test_data_file = 'exp_test.csv'
    train_data_file = 'exp_train.csv'

    # data sets prep
    pseudo_data_size, test_df = load_data(pseudo_data_folder, test_data_file)

    true_label = test_df.label
    yelp_review = test_df.review

    data_stream = []

    for i in range(len(yelp_review)):
        data_stream.append((int(true_label[i]), yelp_review[i]))

    print("batch_inference")
    print('Coming Stream is ready...')
    print('===============================')

    # batch stream set up
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.BATCH)
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

    ds = env.from_collection(collection=data_stream)
    accuracy = batch_inference(ds, test_data_size)
    print(accuracy)

    # print("supervised_model_train")

    # # train model on pseudo data with supervised mode
    # pseudo_data_size, train_df = load_and_augment_data(pseudo_data_folder, train_data_file)
    # train_data_size = len(train_df)

    # supervised_model(parallelism, train_df, train_data_size, pseudo_data_size, PSEUDO_DATA_COLLECTION_THRESHOLD,
    #                  accuracy,
    #                  ACCURACY_THRESHOLD)
