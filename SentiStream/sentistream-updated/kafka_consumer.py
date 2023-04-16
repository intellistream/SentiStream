from kafka import KafkaConsumer

# Create Kafka consumer.
consumer = KafkaConsumer(
    'sentiment-data',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda x: x.decode('utf-8')
)


# Consume message from Kafka topic.
for message in consumer:
    print(message.value.split('|||'))
