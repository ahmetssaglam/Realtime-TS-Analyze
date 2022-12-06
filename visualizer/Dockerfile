FROM python:3.8-slim
RUN apt-get update
RUN apt install -y nano
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . ./
CMD ["python", "app.py"]
