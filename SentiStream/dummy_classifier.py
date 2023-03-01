import logging
import redis
import random

from pyflink.datastream.functions import RuntimeContext, MapFunction
from time import time


def default_confidence(model, data):
    return model.predict_proba(data)

class DummyClassifier(MapFunction):
    def __init__(self):
        self.model = None
        self.data = []
        self.redis = None
        self.time_threshold = 10 * 60  
        self.start_time = time()

    # def open(self, runtime_context: RuntimeContext):
    #     """
    #     upload model here  in runtime
    #     """
    #     self.model = default_model_classifier()  # change to your model
    #     self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

    # def load(self):
    #     """
    #     upload model periodically
    #     """
    #     #  periodic load
    #     if time() - self.start_time >= self.time_threshold:
    #         self.start_time = 0
    #         try:
    #             flag = int(self.redis.get('classifier_update'))
    #             if flag == 1:
    #                 self.model = default_model_classifier()  # change to your model here
    #                 self.redis.set('classifier_update', int(False))
    #             elif flag is None:
    #                 logging.warning('classifier_model update flag not set. Train model before running')
    #         except:
    #             raise ConnectionError('Failed to open redis')

    def get_confidence(self, func=default_confidence):
        """
        :param func:a function(classifier,[vector_mean]) that is expects a classifier
                model and data of mean vectors as input
        :return: an array of confidence for 0 and 1 labels e.g. [0.40,0.60]
        """
        neg_conf = random.random()
        return [neg_conf, 1-neg_conf]

    def map(self, data):
        conf = self.get_confidence()


        pred_conf = max(conf)

        return [data[0], pred_conf, conf.index(pred_conf), data[2], data[1]]
        # for i in range(len(data)):
        #     self.data.append(data[i][1])
        # confidence = self.get_confidence()
        # for i in range(len(data)):
        #     if confidence[i][0] >= confidence[i][1]:
        #         data[i][1] = confidence[i][0]
        #         data[i].insert(2, 0)
        #     else:
        #         data[i][1] = confidence[i][1]
        #         data[i].insert(2, 1)

        # return data
    

def dummy_classifier(ds):
    ds = ds.map(DummyClassifier())
    # ds = ds.map(lambda x: x)
    # ds = ds.map(Supervised_OSA_inference()).filter(lambda i: i != 'collecting')
    # # ds.print()
    # # ds.flat_map(split).print() #data size is uneven due to use of collector
    # ds = ds.map(Classifier()).flat_map(split)
    return ds