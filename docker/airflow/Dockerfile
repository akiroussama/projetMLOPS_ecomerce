ARG AIRFLOW_VERSION=2.10.5
FROM apache/airflow:${AIRFLOW_VERSION}-python3.10

COPY requirements-pipeline.txt /tmp/requirements-pipeline.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" && \
    pip install --no-cache-dir -r /tmp/requirements-pipeline.txt

WORKDIR /opt/airflow/project
