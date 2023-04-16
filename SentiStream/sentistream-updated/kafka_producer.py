# pylint: disable=import-error
# pylint: disable=no-name-in-module

import csv
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

TOPIC_NAME = 'sentiment-data'
NUM_PARTITIONS = 1
REPLICATION_FACTOR = 1

# Initialize KafkaAdminClient with bootstrap servers.
admin_client = KafkaAdminClient(bootstrap_servers='localhost:9092')

# Delete topic if already created.
if TOPIC_NAME in admin_client.list_topics():
    admin_client.delete_topics([TOPIC_NAME])

# Create new topic with desired number of partitions and replication factor.
new_topic = NewTopic(name=TOPIC_NAME, num_partitions=NUM_PARTITIONS,
                     replication_factor=REPLICATION_FACTOR)

# Create the topic.
while True:
    try:
        admin_client.create_topics(new_topics=[new_topic])
        break
    except TopicAlreadyExistsError:
        pass

print(f"{TOPIC_NAME} topic has been created.")


# Create Kafka producer.
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: x.encode('utf-8')
)

# Read CSV file and create data stream.
with open('train.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)

    # use triple pipe to separate label and text as it is unlikely to occur in text.
    for row in reader:
        producer.send(TOPIC_NAME, value=f'{int(row[0]) - 1}|||{row[1]}')
