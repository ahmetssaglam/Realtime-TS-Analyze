from time import time
from functools import wraps
import numpy as np
import pandas as pd
import math
import os

FILE_PREFIX = "raw_data/2nd_test"


def calculate_time(func):
    @wraps(func)
    def inner(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        finish = time()
        print(f'{func.__name__} function completed in {finish - start} seconds.')
        return result

    return inner


@calculate_time
def get_rms(df, column_name):
    result = np.sqrt((df[column_name] ** 2).sum() / len(df[column_name]))
    return result


@calculate_time
def rmsValue(df, column_name):
    square = 0

    arr = df[column_name].tolist()
    array_length = len(arr)

    # Calculate square
    for i in range(0, array_length):
        square += (arr[i] ** 2)

    # Calculate Mean
    mean = (square / float(array_length))

    # Calculate Root
    root = math.sqrt(mean)

    return root


def make_calculation(folder_path, calc_func, calc_name, result_path):
    data_files = os.listdir(folder_path)
    data_files.sort()

    columns = ['bearing_1', 'bearing_2', 'bearing_3', 'bearing_4']

    # init column and date lists
    result_dict = {}
    result_dict["date_list"] = []
    for col in columns:
        result_dict['%s_list' % col] = []

    for data_file in data_files:
        dataset = pd.read_csv(os.path.join(folder_path, data_file), sep='\t')
        dataset.columns = columns

        for column_name in dataset:
            res = calc_func(dataset, column_name)  # make calculation for column
            result_dict['%s_list' % column_name].append(res)  # append result to list in result dict
        result_dict["date_list"].append(pd.to_datetime(data_file, format='%Y.%m.%d.%H.%M.%S'))

    columns.insert(0, "datetime")
    result_df = pd.DataFrame.from_dict(result_dict)
    result_df.columns = columns
    result_df.to_csv(os.path.join(result_path, calc_name + ".csv"), index=False)



if __name__ == '__main__':
    # dataset = pd.read_csv('raw_data/2nd_test/2004.02.12.10.32.39', sep='\t')
    # dataset.columns = ['bearing_1', 'bearing_2', 'bearing_3', 'bearing_4']
    # # print(dataset.head())
    # rms_1 = rmsValue(dataset, 'bearing_1')
    # rms_2 = get_rms(dataset, 'bearing_1')
    # print(f'with numpy => {rms_2}, other => {rms_1}')

    # raw_data_file_names = os.listdir(FILE_PREFIX)
    # raw_data_file_names.sort()
    # print(raw_data_file_names[:5])

    start = time()
    make_calculation(folder_path="raw_data/2nd_test", calc_func=get_rms, calc_name="RMS", result_path="data/2")
    print(time() - start)

