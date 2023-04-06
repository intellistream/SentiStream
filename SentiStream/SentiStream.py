

# TORCH JIT ?
# TODO: FIX CLASS IMBALANCE IN INITIAL TRAINING

import os
import shutil
import pandas as pd

from pyflink.datastream import CheckpointingMode, StreamExecutionEnvironment
from pyflink.datastream.execution_mode import RuntimeExecutionMode

from classifier import classifier
from batch_inferrence import batch_inference
from evaluation import generate_new_label, merged_stream
from supervised_model import supervised_model
from PLStream import unsupervised_stream
from train_model import InitialModelTrain
from utils import load_data

if __name__ == '__main__':
    parallelism = 1

    if os.path.exists('senti_output'):
        shutil.rmtree('senti_output', ignore_errors=False, onerror=None)

    # set train_data as first 1000

    ## -------------------INITIAL TRAINING OF SUPERVISED MODEL------------------- ##

    new_df = pd.read_csv('train.csv', names=['label', 'review'])
    new_df['label'] -= 1

    df = new_df[:1000]

    InitialModelTrain([(int(label), review) for label, review in df.values])

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

    df = new_df[1000:2000]

    ## -------------------GENERATE PSEUDO-LABEL FROM BOTH LEARNING METHODS------------------- ##

    data_stream = [(i, int(label), review) for i, (label, review) in enumerate(df.values)]

    # env.set_parallelism(1)
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)

    ds = env.from_collection(collection=data_stream)

    print("unsupervised stream,classifier and evaluation")

    ds1 = unsupervised_stream(ds)
    ds2 = classifier(ds)

    ds = merged_stream(ds1, ds2)
    generate_new_label(ds)

    # ds.print()

    ## -------------------SUPERVISED MODEL INFERENCE------------------- ##

    print("batch_inference")

    # env.set_parallelism(1)

    env.set_runtime_mode(RuntimeExecutionMode.BATCH)

    acc = batch_inference(ds)

    ## -------------------SUPERVISED MODEL TRAIN-------------------##
    print("supervised_model_train")

    supervised_model(acc, pseudo_data_collection_threshold=0.0, accuracy_threshold=1.0)

    env.execute()
        
