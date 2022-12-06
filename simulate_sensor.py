import json
import os
import pandas as pd
from time import sleep
from kafka import KafkaProducer

FILE_PATH = "simulation_data"
SLEEP_TIME = 1  # in seconds

# kafka configs
KAFKA_HOST = os.environ.get("KAFKA_HOST", "localhost")
KAFKA_PORT = os.environ.get("KAFKA_PORT", "9092")
KAFKA_DATA_TOPIC = os.environ.get("KAFKA_DATA_TOPIC", "data_topic")


def simulate(folder_path=FILE_PATH, sleep_time=SLEEP_TIME, kafka_host=KAFKA_HOST, kafka_port=KAFKA_PORT,
             kafka_topic=KAFKA_DATA_TOPIC):
    # create producer
    KAFKA_SERVER = f'{kafka_host}:{kafka_port}'
    producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER)

    # read raw data
    data_files = os.listdir(folder_path)
    data_files.sort()

    columns = ['bearing_1', 'bearing_2', 'bearing_3', 'bearing_4']

    for data_file in data_files:
        dataset = pd.read_csv(os.path.join(folder_path, data_file), sep='\t')
        dataset.columns = columns

        data_list = dataset['bearing_1'].tolist()
        file_date = pd.to_datetime(data_file, format='%Y.%m.%d.%H.%M.%S')
        file_date_str = file_date.strftime("%Y-%m-%d %H:%M:%S")

        message = {"date": file_date_str,
                   "raw_data": data_list}

        try:
            producer.send(kafka_topic, value=json.dumps(message).encode("utf-8"))
            producer.flush()
            print(f'{file_date} Bearing 1 sent..')
        except Exception as err:
            print(err)

        sleep(sleep_time)

    # close producer
    producer.close()


if __name__ == '__main__':
    # start simulation
    simulate()
