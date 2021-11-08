# Online-Sentiment-Analysis-on-Twitter-Streams
Master Thesis "An Empirical Study of  Online Sentiment Analysis on Twitter Streams " offered by [DIMA](https://www.dima.tu-berlin.de/menue/database_systems_and_information_management_group/?no_cache=1), TU Berlin
* Author: Huilin Wu
* Advisor: [Dr. Shuhao Zhang](https://github.com/ShuhaoZhangTony)

# Time Table
1. Experiments of impact of text length. 3/Nov/2021 -- done.
2. Minimum Word Count experiments Revise. 4/Nov/2021 -- done.
3. Impact of Optimizations. 12/Nov/2021
4. Experimental analysis of violint figures. 15/Nov/2021
5. Draft update. 19/Nov/2021
6. Send to Volker for review. 22/Nov/2021
7. Prepare the demo for DASFAA. 28/Nov/2021
8. Make the submission to VLDB 22. 1/Dec/2021

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

## Quick Guide
### 1. Supervised_OSA
* [Tweets_clean_demo](https://github.com/HuilinWu2/Online-Sentiment-Analysis-on-Twitter-Streams/tree/main/Pyflink_demo/Tweets_clean_demo): Demo for batch-based Tweets preprocessing for Sentiment Analysis on FLink
* [Streaming_demo_Sentiment_Analysis](https://github.com/HuilinWu2/Online-Sentiment-Analysis-on-Twitter-Streams/tree/main/Pyflink_demo/Streaming_demo_Sentiment_Analysis): Demo for stream-based Tweets preprocessing & online Sentiment Analysis model on Flink
### 2. Unsupervised_OSA
* Algorithms of incremental Sentiment Analysis
### 3. Python_small_job
* Python file of various developing note
