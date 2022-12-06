import os
import json
from util import load_json, update_config, add_row_to_df, append_hourly_anomaly, initialize_all_csv, update_all_csv
from ops import load_model, make_prediction, train, is_anomaly, preprocess, write_alarm_to_db, is_type2_alarm
from kafka import KafkaConsumer
import psycopg2
# import pandas as pd

import logging.config

# disable 3rd libraries loggings
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

# logging config
logging.basicConfig(format='%(levelname)s: %(asctime)s - %(message)s', level=logging.INFO)

INPUT_FILE = "bearing1_train_half.csv"

TRAIN_FILE = os.environ.get("TRAIN_FILE", "data/train.csv")
PREDICTION_FILE = os.environ.get("PREDICTION_FILE", "data/prediction.csv")
WEIGHT_FILE = os.environ.get("WEIGHT_FILE", "weights/model_weights.json")
CONFIG_FILE = os.environ.get("CONFIG_FILE", "config.json")
ANOMALY_COUNT_FILE = os.environ.get("ANOMALY_COUNT_FILE", "anomaly_count_list.json")
ALL_DATA_FILE = os.environ.get("ALL_DATA_FILE", "data/all.csv")

KAFKA_HOST = os.environ.get("KAFKA_HOST", "localhost")
KAFKA_PORT = os.environ.get("KAFKA_PORT", "9092")
KAFKA_DATA_TOPIC = os.environ.get("KAFKA_DATA_TOPIC", "data_topic")

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5532")
POSTGRES_DATABASE = os.environ.get("POSTGRES_DATABASE", "ai_db")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "admin")

KAFKA_SERVER = f'{KAFKA_HOST}:{KAFKA_PORT}'

if __name__ == '__main__':

    # make an initial prediction
    model = load_model(load_path=WEIGHT_FILE)
    make_prediction(model=model)

    # initialize temporary df for visualization (contains hourly actual value)
    # temp_df = pd.DataFrame(columns=["ds", "y"])
    initialize_all_csv()

    # control connection errors
    try:
        # create consumer
        consumer = KafkaConsumer(KAFKA_DATA_TOPIC, bootstrap_servers=KAFKA_SERVER,
                                 value_deserializer=lambda x: json.loads(x.decode('utf-8')))
        # database connection
        database_conn = psycopg2.connect(user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST,
                                         port=POSTGRES_PORT, database=POSTGRES_DATABASE)
    except Exception as err:
        raise err

    # connection log messages
    logging.info(f'Connected Database {POSTGRES_DATABASE} at {POSTGRES_HOST}:{POSTGRES_PORT}')
    logging.info(f'Listening Queue {KAFKA_DATA_TOPIC} at {KAFKA_SERVER}')

    # read config
    config_dict = load_json(file_name=CONFIG_FILE)
    count = int(config_dict["count"])
    min_value = float(config_dict["min_value"])
    max_value = float(config_dict["max_value"])
    anomaly_threshold = float(config_dict["anomaly_threshold"])
    alarm_threshold = float(config_dict["alarm_threshold"])
    hourly_anomaly_count = int(config_dict["hourly_anomaly_count"])

    logging.info(f'Configurations read from config.json')

    # consumer listens kafka topic
    for message in consumer:

        # get date and rms value from consumed message
        fetched_date, fetched_value = preprocess(message.value)

        # compare prediction and normalized value
        result_dict = is_anomaly(value=fetched_value, count=count, anomaly_threshold=anomaly_threshold,
                                 alarm_threshold=alarm_threshold, prediction_csv=PREDICTION_FILE)

        # if anomaly detected
        if result_dict["alarm"]:
            # log anomaly
            logging.error(f'ALARM DETECTED! Alarm Time: {fetched_date} -- Alarm Type: 1')

            try:
                bound_dict = {"yhat_lower": result_dict["yhat_lower"], "yhat_upper": result_dict["yhat_upper"]}
                # write anomaly to db
                write_alarm_to_db(db_conn=database_conn, alarm_date=fetched_date, alarm_value=fetched_value,
                                  bound_dict=bound_dict, alarm_type=1)
            except Exception as err:
                logging.error(err)
            # update anomaly count
            hourly_anomaly_count += 1

        elif result_dict["anomaly"]:
            # update anomaly count
            hourly_anomaly_count += 1

        # add new data to csv (train.csv)
        add_row_to_df(date=fetched_date, value=fetched_value, data_df_path=TRAIN_FILE)

        # add new data to all.csv (for visualization)
        update_all_csv(ds=fetched_date, y=fetched_value, yhat_lower=result_dict["yhat_lower"],
                       yhat_upper=result_dict["yhat_upper"], yhat=result_dict["yhat"])

        # # add new data to dataframe for visualization
        # temp_df.loc[len(temp_df.index)] = [fetched_date.replace("/", "-"), fetched_value]

        # update count
        count += 1

        # if pointer has reached to limit (if hour is completed)
        if count >= 6:
            # # visualization
            # visualize_prediction(original_data=temp_df, predictions_data_path=PREDICTION_FILE, all_path=ALL_DATA_FILE)
            # # reset temporary dataframe
            # temp_df = pd.DataFrame(columns=["ds", "y"])

            update_config(hourly_anomaly_count=0, count=0, update_min_max=True, config_path=CONFIG_FILE, df_path=TRAIN_FILE)
            # retrain
            model = train(data_df_path=TRAIN_FILE, save_path=WEIGHT_FILE)
            # prediction
            make_prediction(model=model, file_name=PREDICTION_FILE, freq='10min', period=6)

            # append hourly anomaly
            append_hourly_anomaly(count_list_path=ANOMALY_COUNT_FILE, hourly_anomaly_count=hourly_anomaly_count)

            # control if trend alarm is detected
            trend_alarm = is_type2_alarm(count_list_path=ANOMALY_COUNT_FILE, hour_count_threshold=3, threshold=0.5)
            # if trend alarm is exist, write it to db
            if trend_alarm is not None:
                logging.error(f'ALARM DETECTED! Alarm Time: {fetched_date} -- Alarm Type: 2')
                write_alarm_to_db(db_conn=database_conn, alarm_date=fetched_date, alarm_value=trend_alarm,
                                  bound_dict=None, alarm_type=2)

            # read config
            config_dict = load_json(file_name=CONFIG_FILE)
            count = int(config_dict["count"])
            min_value = float(config_dict["min_value"])
            max_value = float(config_dict["max_value"])
            anomaly_threshold = float(config_dict["anomaly_threshold"])
            alarm_threshold = float(config_dict["alarm_threshold"])
            hourly_anomaly_count = int(config_dict["hourly_anomaly_count"])
        else:
            update_config(hourly_anomaly_count=hourly_anomaly_count, count=count, config_path=CONFIG_FILE)

    # close consumer
    consumer.close()
    # close database connection
    database_conn.close()
