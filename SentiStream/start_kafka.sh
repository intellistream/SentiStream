#!/bin/bash

# Delete running servers.
kafka/bin/kafka-server-stop.sh &
sleep 10
kafka/bin/zookeeper-server-stop.sh &
sleep 10

# Start Zookeeper and wait for it to load.
kafka/bin/zookeeper-server-start.sh kafka/config/zookeeper.properties &
sleep 10

# Start Kafka and wait for it to load.
kafka/bin/kafka-server-start.sh kafka/config/server.properties &
sleep 10