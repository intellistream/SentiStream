# SentiStream: Towards Online Sentiment Learning of Massive Data Streams

## Motivation
* Sentiment classification over a large number of opinions, like tweets or product reviews, has numerous practical applications, such as public opinion monitoring, customer support management, and event prediction. However, existing sentiment classification techniques cannot effectively handle continuous evolving data streams in real-time as they typically assume that a large number of datasets have been collected and labelled beforehand. 
* we introduce SentiStream, an online sentiment learning system that learns incrementally, trains without labels, and scales almost linearly.

## Environment Requirements
1. Flink v1.12
2. Scala v2.11
3. Python 3.7
4. Java 8
5. Kafka 2.13
6. Redis server v4.0.9

## DataSource
### Tweets
* 1.6 million labeled Tweets:
* [Sentiment140](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
### Yelp Reviews
* 280,000 training and 19,000 test samples in each polarity
* [Yelp Review Polarity](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz)
### Amazon Reviews
* 1,800,000 training and 200,000 testing samples in each polarity
* [Amazon product review polarity](https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz)

* Dataset quick access in https://course.fast.ai/datasets#nlp

## Quick Guide
### 1. Supervised_OSA
* [Tweets_clean_demo](https://github.com/HuilinWu2/Online-Sentiment-Analysis-on-Twitter-Streams/tree/main/Pyflink_demo/Tweets_clean_demo): Demo for batch-based Tweets preprocessing for Sentiment Analysis on Flink
* [Streaming_demo_Sentiment_Analysis](https://github.com/HuilinWu2/Online-Sentiment-Analysis-on-Twitter-Streams/tree/main/Pyflink_demo/Streaming_demo_Sentiment_Analysis): Demo for stream-based Tweets preprocessing & online Sentiment Analysis model on Flink
### 2. Unsupervised_OSA
* Algorithms of incremental Sentiment Analysis
### 3. Python_small_job
* Python file of various developing note
