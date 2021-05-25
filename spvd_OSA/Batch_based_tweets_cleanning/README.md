# Demo for batch-based tweets cleanning using udf in pyflink

## Description
Input:200 raw tweets --> tweets.text
output: tweets that are cleaned for SA --> tweets_clean.csv


## Environment Requirement:
1. Python 3.7+
2. Flink v1.11

## Quick start:
1. start flink: `./bin/start-cluster.sh`
2. Run in terminal: `python demo_tweets_clean.py`
