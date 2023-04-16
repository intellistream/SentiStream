#!/bin/bash

# Delete running servers.
kafka/bin/kafka-server-stop.sh &
sleep 10
kafka/bin/zookeeper-server-stop.sh &
sleep 10

# Start Zookeeper.
kafka/bin/zookeeper-server-start.sh kafka/config/zookeeper.properties &
sleep 10

# Start Kafka.
kafka/bin/kafka-server-start.sh kafka/config/server.properties &
sleep 10

# Create topic for streaming, (delete if already exists).
kafka/bin/kafka-topics.sh --delete --topic sentiment-data --bootstrap-server localhost:9092
kafka/bin/kafka-topics.sh --create --topic sentiment-data --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1