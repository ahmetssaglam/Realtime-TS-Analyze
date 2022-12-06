# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Output, Input


def visualize_prediction(all_path="data/all.csv", alarm_threshold=0.5, anomaly_threshold=0.1):
    """
    visualizes actual data, anomaly and alarm bounds
    :param all_path: path of all.csv that contains actual (real) and predicted data together
    :param alarm_threshold: alarm threshold
    :param anomaly_threshold: anomaly threshold
    :return:
    """

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
                'width': .85
            }
        },
        name='Actual'
    )

    # create layout
    layout = go.Layout(
        yaxis={
            'title': "Bearing Vibration",
            "color": "#e0e0e0",
            "titlefont": {"family": "Arial Black"},
            "showgrid": True,
            "gridwidth": 0.001,
            "gridcolor": "#e8dcdc",
            "zeroline": True,
            "zerolinewidth": 1,
            "zerolinecolor": "black"
        },
        hovermode='closest',
        xaxis={
            'title': "Datetime",
            "color": "#e0e0e0",
            "titlefont": {"family": "Arial Black"},
            "showgrid": True,
            "gridwidth": 0.001,
            "gridcolor": "#e8dcdc",
            # "zeroline": True,
            # "zerolinewidth": 3,
            # "zerolinecolor": "black"
        },
        margin={
            't': 50,
            'b': 50,
            'l': 60,
            'r': 10
        },
        legend={
            'bgcolor': '#e4f1fe',
        },
        font={"color": "black"},
        paper_bgcolor="#222831",
        plot_bgcolor="#e4f1fe",
        uirevision="don't change",
    )

    # gather all scatters
    data = [anomaly_lower_scatter, anomaly_upper_scatter, alarm_upper_scatter, alarm_lower_scatter, yhat_scatter,
            actual_scatter]
    # create figure
    fig = dict(data=data, layout=layout)



    return fig


app = dash.Dash(__name__, title="Bearing Analyze", update_title="Bearing Analyze..")


@app.callback(Output("bearing-graph", "figure"), Input("interval-component", "n_intervals"))
def update_graph(n):
    return visualize_prediction()


header = html.H2(children='Real Time Bearing Data', style={'text-align': 'center', "color": "#e0e0e0", "font-family": "Arial Black"})
interval = dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
graph = dcc.Graph(id='bearing-graph', style={"textAlign": "center", "marginTop": "50px", "height": "800px", "marginBottom": "20px"}, figure=visualize_prediction())

app.layout = html.Div(children=[
    header,
    interval,
    graph
])


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=7755, debug=True)
    app.run(host='0.0.0.0', port=7755)
