import ujson

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def normalize_list(values):
    """
    normalize values between 0 - 1 in a list
    :param values: value list
    :return: normalized list
    """
    norm = [(float(i) - min(values)) / (max(values) - min(values)) for i in values]
    return norm


def normalize_value(min_value, max_value, value):
    """
    normalize single value according to min and max value
    :param min_value: minimum value
    :param max_value: maximum value
    :param value: value to be normalized
    :return: normalized value
    """
    return (float(value) - min_value) / (max_value - min_value)


def normalize_column(dataframe, column_name='y'):
    """
    normalize column of dataframe
    :param dataframe: dataframe that contains the column to normalize
    :param column_name: the name of the column to be normalized.
    :return: dataframe that contains normalized column
    """
    dataframe[column_name] = MinMaxScaler().fit_transform(np.array(dataframe[column_name]).reshape(-1, 1))
    return dataframe


def save_json(conf_dict, file_name="config.json"):
    """
    saves the dict object to file system in json format
    :param conf_dict: dict to be saved
    :param file_name: saved file name
    :return:
    """
    with open(file_name, "w") as f:
        ujson.dump(conf_dict, f, indent=2)
        f.close()
    # logging.info(f'Config saved to {file_name}')


def load_json(file_name="config.json"):
    """
    loads the json file as a dict object from file system
    :param file_name: file name that contains the json
    :return: loaded dict object
    """
    with open(file_name, "r") as f:
        conf_dict = ujson.load(f)
        f.close()
    # logging.info(f'Config fetched from {file_name}')
    return conf_dict


def update_config(hourly_anomaly_count, count=0, update_min_max=False, config_path="config.json",
                  df_path="data/train.csv"):
    """
    updates config file
    :param hourly_anomaly_count: anomaly count of that hour
    :param count: how many times value that come from stream (prediction.csv row pointer)
    :param update_min_max: if true, update min-max value in config.json
    :param config_path: config file path
    :param df_path: train.csv (original train data) file path
    :return:
    """
    # read config
    config_dict = load_json(file_name=config_path)

    # update config
    config_dict["count"] = count
    config_dict["hourly_anomaly_count"] = hourly_anomaly_count

    if update_min_max:
        # read dataframe
        df = pd.read_csv(df_path)

        config_dict["max_value"] = df['y'].max()
        config_dict["min_value"] = df['y'].min()

    save_json(conf_dict=config_dict, file_name=config_path)


def add_row_to_df(date, value, data_df_path="data/train.csv"):
    """
    add row to existing train dataframe and save dataframe to file
    :param date: date value for ds column
    :param value: value for y column
    :param data_df_path: train.csv file path
    :return:
    """
    # read csv
    data_df = pd.read_csv(data_df_path)

    # append the row
    data_df.loc[len(data_df.index)] = [date, value]

    # save csv
    data_df.to_csv(data_df_path, index=False)


def get_rms(value_list):
    """
    returns rms value of give list
    :param value_list: list of values
    :return: rms value in np.float64 format
    """
    value_array = np.asarray(value_list)
    result = np.sqrt((value_array ** 2).sum() / len(value_array))
    return result


def append_hourly_anomaly(count_list_path, hourly_anomaly_count):
    """
    append anomaly value to the list that exists in json
    :param count_list_path: path of json file
    :param hourly_anomaly_count: value (hourly anomaly count) to be appended
    :return:
    """
    # get json from file
    anomaly_count_json = load_json(file_name=count_list_path)
    # append value to the list
    anomaly_count_json["hourly_anomalies"].append(hourly_anomaly_count)
    # save json to the file
    save_json(file_name=count_list_path, conf_dict=anomaly_count_json)


def initialize_all_csv(all_csv_path="data/all.csv"):
    df = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper", "y"])
    df.to_csv(all_csv_path, index=False)


def update_all_csv(ds, y, yhat, yhat_lower, yhat_upper, all_csv_path="data/all.csv"):

    # read csv
    all_df = pd.read_csv(all_csv_path)

    # append the row
    all_df.loc[len(all_df.index)] = [ds, yhat, yhat_lower, yhat_upper, y]

    # save csv
    all_df.to_csv(all_csv_path, index=False)

# split df to half

# dfs = np.array_split(df, 2)
# train_df = dfs[0]
# train_df.to_csv("bearing1_train_half.csv", index=False)
# df = train_df
# exit(2)
