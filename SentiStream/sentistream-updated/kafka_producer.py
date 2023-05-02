# pylint: disable=import-error
# pylint: disable=no-name-in-module

import csv

from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

import config

NUM_PARTITIONS = 1
REPLICATION_FACTOR = 1

# Initialize KafkaAdminClient with bootstrap servers.
admin_client = KafkaAdminClient(bootstrap_servers=config.BOOTSTRAP_SERVER)

# Delete topic if already created.
if config.KAFKA_TOPIC in admin_client.list_topics():
    admin_client.delete_topics([config.KAFKA_TOPIC])

# Create new topic with desired number of partitions and replication factor.
new_topic = NewTopic(name=config.KAFKA_TOPIC, num_partitions=NUM_PARTITIONS,
                     replication_factor=REPLICATION_FACTOR)

# Create the topic.
while True:
    try:
        admin_client.create_topics(new_topics=[new_topic])
        break
    except TopicAlreadyExistsError:
        pass

print(f"{config.KAFKA_TOPIC} topic has been created.")


# Create Kafka producer.
producer = KafkaProducer(
    bootstrap_servers=config.BOOTSTRAP_SERVER,
    value_serializer=lambda x: x.encode('utf-8')
)

# Read CSV file and create data stream.
with open(config.DATA, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)

    ss_train = []

    # use triple pipe to separate label and text as it is unlikely to occur in text.
    for row in reader:
        producer.send(config.KAFKA_TOPIC,
                      value=f'{int(row[0]) - 1}|||{str(row[1])}')
