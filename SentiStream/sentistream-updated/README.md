# SentiStream

## Prerequisites

- Python (>=3.10)
- Apache Kafka
- Required Python libraries (see `requirements.txt`) 

### Install Kafka

Download latest stable release
```
wget https://downloads.apache.org/kafka/3.4.0/kafka_2.13-3.4.0.tgz
```

Extract
```
tar -xzvf kafka_2.13-3.4.0.tgz && mv kafka_2.13-3.4.0 kafka
```

### Install Pythin libraries

Install libraries
```
pip install -r requirements.txt
```

`apache-flink` installs outdated `numpy` which have dependecy issue with other libs, upgrading won't affect flink performance.

```
pip install --upgrade numpy
```

## Usage


Start Kafka server
```
sh start_kafka.sh
```

Create producer
```
python3 kafka_producer.py
```

Start SentiStream
```
python3 main.py
```

> NOTE: Change parameters from `config.py` to train with different word vector or classifier algorithms.



###### ----------------------
## DONT SINK PSEUDO LABELS...


## MOVE TO CONFLUENT-KAFKA

## WHAT ABT MAX SENT, WORD LEN???

## CHECK FOR LAST BATCH IN STREAM

# TO REM
## WHY DO WE HAVE BATCH INF?, EVEN IF 90% ACC, IT MAY HAVE SOME NEW VOCABS TO LEARN??? 

### IS TEMPORAL TREAD DETECTION REQ??
### PSEUDO_LABEL -> THRESHOLDDDDS


# SHAREM MEM ARRAY TO STORE ALL O/P BEFORE TRAIN-------------



### NUM WORKERS FOR WV - PLSTREAM, CLASSIFIER ---- find best for each batchsizw



### THINGS TO NOTE;;;

- plstream uses trained word vec from initial training
- han uses dynamic expansion of embedding matrix for continual training
- took avg of cosine sim