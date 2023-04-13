# pylint: disable=import-error

# US WORD_VECTOR_NAME = 'plstream-wv.model'
# SS WORD_VECTOR_NAME = 'ssl-wv.model'
# CLASSIFIER_NAME = 'clf.pth'

# TORCH JIT ?
# TODO: FIX CLASS IMBALANCE IN INITIAL TRAINING - DONE - REWATCH
# TODO: UPDATE HAN PREPROCESSING FROM 08a914a COMMIT


from supervised import TrainModel

from time import time

# Initial training of classifier
# Train word vector {Word2Vec or FastText} on {nrows=1000} data and get word embeddings, then use
# that embedding to train NN sentiment classifier {ANN or HAN}

# start = time()

TrainModel(word_vector_algo='Word2Vec', ssl_model='HAN', init=True, nrows=1000)

# print(time() - start)
