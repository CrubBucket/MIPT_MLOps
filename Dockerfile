FROM ghcr.io/mlflow/mlflow:v2.13.0

WORKDIR /app

ADD . /app

RUN pip install --upgrade pip && pip install --default-timeout=1000 -r requirements.txt

RUN mlflow server --host localhost --port 5000