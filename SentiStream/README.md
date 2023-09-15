# SentiStream: A Co-Training Framework for Adaptive Online Sentiment Analysis in Evolving Data Streams
​
Online sentiment analysis is crucial for social media monitoring, customer feedback analysis, and online reputation management. Existing methods struggle with evolving data streams due to their reliance on pre-existing labeled datasets. This research introduces SentiStream, a co-training framework that enables efficient sentiment analysis in dynamic data streams. By leveraging unsupervised, semi-supervised, and stream merge modules, SentiStream adapts to changing data landscapes and improves analysis accuracy. The research focuses on real-world applications and aims to improve both the accuracy and computational efficiency of sentiment analysis in dynamic data streams.
​
## Prerequisites
​
- Python (v3.10 or higher)
- Apache Kafka
- Required Python libraries (listed in `requirements.txt`) 
​
​
## Installation
​
1. Download the latest stable release of Apache Kafka:
```
wget https://downloads.apache.org/kafka/3.4.0/kafka_2.13-3.4.0.tgz
```
​
2. Extract the downloaded file and rename the extracted folder to 'kafka':
```
tar -xzvf kafka_2.13-3.4.0.tgz && mv kafka_2.13-3.4.0 kafka
```
​
3. Install the necessary Python libraries:
​
```
pip install -r requirements.txt
```
​
## Datasets
​
The repository includes three distinct datasets to assess its performancein different scenarios for evaluation. These datasets can be used to test the performance of the sentiment analysis models implemented in the code.
​
- To simulate concept drift, 3 different domain datasets have combined combined together.
    - [Yelp]((https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz)) - Randomly sampled dataset of 80,000 data points from Yelp reviews.
    - [LMRD](https://ai.stanford.edu/~amaas/data/sentiment/) - Combined dataset of the Large Movie Review Dataset training and test sets.
    - [SST-2](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip) -  Combined dataset of the Stanford Sentiment Treebank training and validation sets. 
​
​
- [Amzon Fashion Dataset](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/AMAZON_FASHION.json.gz) - Chronologically ordered customer reviews covering the period from 2010 to 2018 from the Amazon e-commerce platform specifically related to fashion products.
​
- [Sentiment140](https://docs.google.com/file/d/0B04GJPshIjmPRnZManQwWEdTZjg/edit?pli=1&resourcekey=0-betyQkEmWZgp8z0DFxWsHw) - 1.6 million tweets from 6 Apr 2009 to 25 Jun 2009 from Twitter to assess temporal context.
​
​
## Evaluation
​
To evaluate the performance of the sentiment analysis models, follow these steps:
​
1. Generate dataset for evaluation:
```
python3 data.py
```
​
2. Start Kafka server:
```
sh start_kafka.sh
```
​
3. Run tests with desired configuration:
```
python3 test.py
```
​
 *NOTE: For SentiStream, pre-trained models trained on respective training data are available in the `trained_models` directory* 
​
> To test other baselines, you can run `other_tests.py`. For the Weakly supervised approach, run `other_exp/cl-wstc/main.py`.
