# SentiStream

## Prerequisites

- Python (>=3.10)
- Apache Kafka
- Required Python libraries (see `requirements.txt`) 

## Datasets

- Yelp
- IMDb
- SST-2

### Install Kafka

Download latest stable release
```
wget https://downloads.apache.org/kafka/3.4.0/kafka_2.13-3.4.0.tgz
```

Extract
```
tar -xzvf kafka_2.13-3.4.0.tgz && mv kafka_2.13-3.4.0 kafka
```

### Install Python libraries

Install libraries
```
pip install -r requirements.txt
```

### For Evaluation

1. Generate dataset for evaluation
```
python3 data.py
```

2. Start Kafka server
```
sh start_kafka.sh
```

3. Run test - with chosen tests
```
python3 test.py
```

 *NOTE: For SentiStream, models trained on training data is available on `trained_models` for each dataset* 

> To test BERT, Self learning, run `other_tests.py` 


