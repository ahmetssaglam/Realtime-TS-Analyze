## TIME SERIES ANOMALY DETECTION ON BEARING DATASET
In this project, the time series model was trained using the [**NASA Bearing Dataset**](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset)
and anomaly detection was performed on this dataset.

Dataset consist of **3** sub dataset and each of them consists of individual files that are 1-second
vibration signal snapshots recorded at specific intervals.

In this project, the second subset was selected. The second subset contains **984** files.
Each file consists of **20480** lines containing values for **4** different bearings.

### PREPROCESS
Only the values of the first bearing are used in the project.
**RMS** (root mean square) of 20480 values in each file is taken. Thus, 20480 values have decreased to **1** value.
All RMS values added to **bearing1_train.csv** file with time value.

The time series model is trained with 1 RMS value obtained for each time period.
Half of the values of bearing1_train.csv is used for model training.
The data to be used in model training is written to **train.csv**.

### TRAIN AND PREDICTION
Initial model was trained with data in the *train.csv* file. Model weights is saved to *weights/model_weights.json*.
Once system is up, model will train himself
continuously every 6 data (every 1 hour data, each data in one-hour period comes with 10 minute frequency).
After each training, model weights saved to *weights/model_weights.json* and new data inserted to the *train.csv*.

After each hour model predicts next hour. Data comes in every 10 minutes and algorithm compares real data and predictions
in every 10 minutes.

Alarm and Anomaly Bounds with Actual Data:
| ![Predictions](/pictures/prediction.png?raw=true "Predictions") |
|:--:|
| <b>Predictions and Real Data</b>|


#### ALARM TYPES
System has 2 type of alarms: **Type 1** and **Type 2**.
If new coming data is increases or decreases suddenly (above or below prediction with %50 threshold),
system will create Type 1 alarm. On the other hand, if more than %50 (>= 9 data point) of data that came in last 3 hours (18 data point) is anomaly,
then system will create Type 2 alarm.

### SIMULATION
To operate the system, first of all, all containers must be up.
To up containers run this command in the main directory of the project:
```
docker-compose up -d
```
If the project is running for the **first time**, run this command instead of above:
```
docker-compose up -d --build
```

| ![docker-compose up](/pictures/compose_up.png?raw=true "Containers") |
|:--:|
| <b>Containers</b>|

After all containers stand up, the simulation can be started with the following command:
```
python3 simulate_sensor.py
```

| ![Simulation](/pictures/simulate_sensor.png?raw=true "Simulation") |
|:--:|
| <b>Simulation</b>|

To view the logs run this command:
```
docker logs ai-module --follow
```

| ![Logs](/pictures/docker_logs.png?raw=true "Simulation Logs") |
|:--:|
| ![Logs](/pictures/docker_logs2.png?raw=true "Simulation Logs") |
|:--:|
| <b>Simulation Logs</b>|

Simulation data can be manipulated in *simulation_data/* folder.

The Queue Visualizer can be accessed at <ins>localhost:9000</ins>.

| ![Kafka Queue](/pictures/kafdrop1.png?raw=true "Simulation Data in the Kafka Topic") |
|:--:|
| ![Kafka Queue2](/pictures/kafdrop2.png?raw=true "Simulation Data in the Kafka Topic") |
| <b>Simulation Data in the Kafka Topic</b>|


PgAdmin can be accessed at <ins>localhost:5533</ins>.
| ![Postgres Database](/pictures/pgadmin.png?raw=true "PostgreSQL alarms Database") |
|:--:|
| <b>PostgreSQL alarms Database</b>|

Execute following command in the main directory of the project to stop the project and remove containers:
```
docker-compose down --volumes
```
After project is stopped, Kafka and PostgreSQL
datas can be removed with deletion of *kafka-data/* and *postgres-data/* folders.

