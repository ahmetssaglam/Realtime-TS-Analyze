import os
import json
from kafka import KafkaConsumer


KAFKA_HOST = os.environ.get("KAFKA_HOST", "localhost")
KAFKA_PORT = os.environ.get("KAFKA_PORT", "9092")
KAFKA_DATA_TOPIC = os.environ.get("KAFKA_DATA_TOPIC", "data_topic")

KAFKA_SERVER = f'{KAFKA_HOST}:{KAFKA_PORT}'
# , auto_offset_reset='earliest'

if __name__ == '__main__':

    # create consumer
    consumer = KafkaConsumer(KAFKA_DATA_TOPIC, bootstrap_servers=KAFKA_SERVER,
                             value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    print(f'listening ==> {KAFKA_SERVER}')

    # consumer listens kafka topic
    for message in consumer:
        value = message.value
        print("GELDI")
        print(type(value))
        print(value)

    # close consumer
    consumer.close()
