CREATE USER akcay;
CREATE DATABASE ai_db;
GRANT ALL PRIVILEGES ON DATABASE ai_db TO akcay;

\c ai_db;
-- CREATE SCHEMA ts-project;

-- CREATE TABLE ts-project.alarms (
CREATE TABLE alarms (
    id SERIAL NOT NULL PRIMARY KEY,
    alarm_date TIMESTAMP NOT NULL,
    alarm_value FLOAT,
    predicted_lower FLOAT,
    predicted_upper FLOAT,
    alarm_type INT NOT NULL,
    UNIQUE (alarm_date, alarm_type)
);