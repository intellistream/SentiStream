# pylint: disable=import-error

import csv
from kafka import KafkaProducer


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
        producer.send('sentiment-data', value='|||'.join(row))
