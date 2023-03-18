FROM  --platform=linux/amd64 python:3.9-slim-buster

WORKDIR /opt/ml/processing

COPY requirements.txt /opt/ml/processing/
RUN pip3 install -r requirements.txt

COPY evaluate.py /opt/ml/processing/

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]