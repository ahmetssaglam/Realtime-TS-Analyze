import pandas as pd
from prophet.serialize import model_to_json, model_from_json
from prophet import Prophet
from util import get_rms, load_json
from psycopg2.errors import UniqueViolation
import plotly.graph_objs as go
import plotly.io as pio
import os


def train(data_df_path="data/train.csv", save_path="weights/model_weights.json"):
    """
    loads data from file and train ts model with data and saves model to the file system
    :param data_df_path: data file path
    :param save_path: save path of trained ts model
    :return: trained ts model
    """
    # load data from file
    dataframe = pd.read_csv(data_df_path)
    # # normalize column before training
    # dataframe = normalize_column(dataframe=dataframe, column_name='y')
    # dataframe["y"] = dataframe["y"] * 100

    # crate model and train
    model = Prophet()
    model.fit(dataframe)

    # save model to file
    save_model(model=model, save_path=save_path)

    return model


def is_anomaly(value, count, anomaly_threshold, alarm_threshold, prediction_csv="data/prediction.csv"):
    """
    controls for anomalies and sudden alarms in the incoming data.
    :param value: incoming data
    :param count: prediction.csv row pointer
    :param anomaly_threshold: anomaly threshold
    :param alarm_threshold: alarm threshold
    :param prediction_csv: csv file that contains future predictions
    :return: return true and bounds in dict if anomaly detected else return false
    """
    # read predictions
    prediction_df = pd.read_csv(prediction_csv)

    # get yhat_lower and yhat_upper value
    yhat_lower, yhat_upper, yhat = prediction_df.loc[count, "yhat_lower"], prediction_df.loc[count, "yhat_upper"], prediction_df.loc[count, "yhat"]

    yhat_lower_anomaly = yhat_lower - (yhat_lower * anomaly_threshold)
    yhat_upper_anomaly = yhat_upper + (yhat_upper * anomaly_threshold)

    yhat_lower_alarm = yhat_lower - (yhat_lower * alarm_threshold)
    yhat_upper_alarm = yhat_upper + (yhat_upper * alarm_threshold)

    alarm = False
    anomaly = False

    # control if alarm exists
    if value <= yhat_lower_alarm or value >= yhat_upper_alarm:
        alarm, anomaly = True, True
    # control if anomaly exists
    elif value <= yhat_lower_anomaly or value >= yhat_upper_anomaly:
        anomaly = True
    # return data
    return {"alarm": alarm, "anomaly": anomaly, "yhat_lower": yhat_lower, "yhat_upper": yhat_upper, "yhat": yhat}


def is_type2_alarm(count_list_path, hour_count_threshold=3, threshold=0.5):
    """
    controls if anomalies is in trend of rise or fall
    :param count_list_path: json path that contains anomaly count list
    :param hour_count_threshold: the number of hours to be used to catch the trend
    :param threshold: the product of the sum of the anomaly scores to decide whether there is a trend
    :return: true and alarm value if alarm exists, else false
    """
    # get anomaly count list from json
    anomaly_count_list = load_json(file_name=count_list_path)["hourly_anomalies"]

    # if not enough anomaly counts return false
    if len(anomaly_count_list) < hour_count_threshold:
        return None
    else:
        # get anomaly counts of last desired hours
        last_hours = anomaly_count_list[-hour_count_threshold:]
        # sum the anomaly counts of last desired hours
        value = sum(last_hours)
        # if sum is above or equal the calculated threshold
        if value >= (hour_count_threshold * 6 * threshold):
            return value
        return None


def make_prediction(model, file_name="data/prediction.csv", freq='10min', period=6):
    """
    This function makes a prediction and saves results to the .csv file
    :param model: ts model that used for making predictions
    :param file_name: file name that prediction results be saved
    :param freq: time frequency, predictions is made every freq value
    :param period: value showing how many freq value forecasts will be made on time
    :return:
    """
    # prepare future dataframe
    future = model.make_future_dataframe(freq=freq, periods=period)

    # make prediction with future time data
    forecast = model.predict(future)
    prediction = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period)

    # save predictions to csv
    prediction.to_csv(file_name, index=False)


def save_model(model, save_path="weights/model_weights.json"):
    """
    saves the time series model in a json format to the file system
    :param model: ts model
    :param save_path: save path
    :return:
    """
    with open(save_path, 'w') as f:
        f.write(model_to_json(model))  # Save model
        f.close()
    # logging.info(f'Model saved to {save_path}')


def load_model(load_path="weights/model_weights.json"):
    """
    loads the time series model from file system
    :param load_path: load path
    :return: ts model
    """
    with open(load_path, 'r') as f:
        model = model_from_json(f.read())  # Load model
        f.close()
    # logging.info(f'Model loaded from {load_path}')
    return model


def preprocess(consumer_message):
    """
    returns date and rms value of given list in a dict
    :param consumer_message: dict object that contains date and raw_data value
    :return: date string and rms value of list
    """
    message_date = consumer_message['date']
    rms_value = get_rms(consumer_message['raw_data'])
    return message_date, rms_value


def write_alarm_to_db(db_conn, alarm_date, alarm_value, alarm_type, bound_dict=None):
    """
    writes detected anomaly to database
    :param db_conn: database connection
    :param alarm_date: date of alarm
    :param alarm_value: alarm value
    :param alarm_type: alarm type 1 = sudden fall or rise, 0 = trend
    :param bound_dict: prediction bounds (bounds for value that predicted)
    :return:
    """
    # create cursor and sql query
    try:
        # sql query that inserts alarm to db
        sql_query = "INSERT INTO alarms (alarm_date, alarm_value, predicted_lower, predicted_upper, alarm_type) VALUES (%s, %s, %s, %s, %s);"
        # crate cursor
        cursor = db_conn.cursor()
    except Exception as err:
        raise err

    # insert anomaly to db
    try:
        # execute sql query
        if bound_dict is None:
            cursor.execute(sql_query, (alarm_date, alarm_value, None, None, alarm_type))
        else:
            cursor.execute(sql_query, (alarm_date, alarm_value, bound_dict["yhat_lower"], bound_dict["yhat_upper"], alarm_type))
        # commit changes via connection
        db_conn.commit()

    # control if anomaly date is already exists in database
    except UniqueViolation as err:
        # close cursor
        cursor.close()
        # revert query
        db_conn.rollback()
        raise err

    except Exception as err:
        # close cursor
        cursor.close()
        raise err


def visualize_prediction(original_data, predictions_data_path, all_path, save_path="prediction_results", alarm_threshold=0.5,
                         anomaly_threshold=0.1):
    """
    visualizes actual data, anomaly and alarm bounds
    :param original_data: dataframe that contains real data
    :param predictions_data_path: file path that contains hourly predictions
    :param save_path: save path of created figure
    :param all_path: path of all.csv that contains actual (real) and predicted data together
    :param alarm_threshold: alarm threshold
    :param anomaly_threshold: anomaly threshold
    :return:
    """
    # read predictions
    predictions = pd.read_csv(predictions_data_path)

    # image_name = original_data["ds"].iloc[0]

    # change data type of ds columns of original data dataframe and predictions dataframe as datetime
    original_data["ds"] = pd.to_datetime(original_data["ds"], infer_datetime_format=True)
    predictions["ds"] = pd.to_datetime(predictions["ds"], infer_datetime_format=True)

    # merge real data and predictions
    final_df = predictions.merge(original_data, on="ds", how='inner')

    # if .csv file that contains original data and predictions is exists
    if os.path.isfile(all_path):
        # append new data
        final_df.to_csv(all_path, mode="a", index=False, header=False)
    else:
        # create file from scratch
        final_df.to_csv(all_path, index=False)

    # read all data (with predictions)
    final_df = pd.read_csv(all_path)

    # get yhat_upper and yhat_lower values of all rows
    yhat_upper = final_df['yhat_upper']
    yhat_lower = final_df['yhat_lower']

    # create scatter for yhat
    yhat_scatter = go.Scatter(
        x=final_df['ds'],
        y=final_df['yhat'],
        mode='lines',
        marker={
            'color': '#3bbed7'
        },
        line={
            'width': 3
        },
        name='Forecast',
    )

    # create scatter for lower anomaly bound
    anomaly_lower_scatter = go.Scatter(
        x=final_df['ds'],
        y=yhat_lower - (yhat_lower * anomaly_threshold),
        marker={
            'color': 'rgba(0,0,0,0)'
        },
        showlegend=False,
        hoverinfo='none',
    )

    # create scatter for upper anomaly bound
    anomaly_upper_scatter = go.Scatter(
        x=final_df['ds'],
        y=yhat_upper + (yhat_upper * anomaly_threshold),
        fill='tonexty',
        fillcolor='rgba(152, 255, 121,.7)',
        name='Confidence',
        hoverinfo='none',
        mode='none'
    )

    # create scatter for lower alarm bound
    alarm_lower_scatter = go.Scatter(
        x=final_df['ds'],
        y=yhat_lower - (yhat_lower * alarm_threshold),
        mode='lines',
        marker={
            'color': '#FF0000'
        },
        line={
            'width': 3
        },
        name='Lower Bound',
    )

    # create scatter for upper alarm bound
    alarm_upper_scatter = go.Scatter(
        x=final_df['ds'],
        y=yhat_upper + (yhat_upper * alarm_threshold),
        mode='lines',
        marker={
            'color': '#FF0000'
        },
        line={
            'width': 3
        },
        name='Upper Bound',
    )

    # create scatter for actual values
    actual_scatter = go.Scatter(
        x=final_df["ds"],
        y=final_df["y"],
        mode='markers',
        marker={
            'color': '#fffaef',
            'size': 4,
            'line': {
                'color': '#000000',
                'width': .75
            }
        },
        name='Actual'
    )

    # create layout
    layout = go.Layout(
        yaxis={
            'title': "y"
        },
        hovermode='x',
        xaxis={
            'title': "hour"
        },
        margin={
            't': 20,
            'b': 50,
            'l': 60,
            'r': 10
        },
        legend={
            'bgcolor': 'rgba(0,0,0,0)'
        }
    )

    # gather all scatters
    data = [anomaly_lower_scatter, anomaly_upper_scatter, alarm_upper_scatter, alarm_lower_scatter, yhat_scatter,
            actual_scatter]
    # create figure
    fig = dict(data=data, layout=layout)
    # write figure to .png file
    pio.write_image(fig=fig, file=f"{save_path}/prediction.png", scale=2.0)
