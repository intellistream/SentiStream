# SentiStream: Towards Online Sentiment Learning of Massive Data Streams

## Motivation
* Most existing studies regarding Sentiment Analysis are based on offline batch-based learning mechanisms. Meanwhile, many stream processing systems have been proposed, but they are not specifically designed for online learning tasks, such as online Sentiment Analysis. As a result, it still remains an open and challenging question of how to efficiently perform Sentiment Analysis for real-time streaming data, e.g., ongoing Twitter Streams.
* The goal of this thesis is to empirically evaluate various online algorithms for Sentiment Analysis on Twitter Streams by implementing them on DSPS (Data Stream Processing System) for practical application.

## Environment Requirement
1. Flink v1.12
2. Scala v2.11
3. Python 3.7
4. Java 8
5. Kafka 2.13
6. Redis server v4.0.9

## DataSource
1.6 million labeled Tweets:
[Sentiment140](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
280,000 training and 19,000 test samples in each polarity
[Yelp Review Polarity](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz)
1,800,000 training and 200,000 testing samples in each polarity
[Amazon product review polarity](https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz)

*Dataset quick access in https://course.fast.ai/datasets#nlp

## Quick Guide
### 1. Supervised_OSA
* [Tweets_clean_demo](https://github.com/HuilinWu2/Online-Sentiment-Analysis-on-Twitter-Streams/tree/main/Pyflink_demo/Tweets_clean_demo): Demo for batch-based Tweets preprocessing for Sentiment Analysis on Flink
* [Streaming_demo_Sentiment_Analysis](https://github.com/HuilinWu2/Online-Sentiment-Analysis-on-Twitter-Streams/tree/main/Pyflink_demo/Streaming_demo_Sentiment_Analysis): Demo for stream-based Tweets preprocessing & online Sentiment Analysis model on Flink
### 2. Unsupervised_OSA
* Algorithms of incremental Sentiment Analysis
### 3. Python_small_job
* Python file of various developing note
