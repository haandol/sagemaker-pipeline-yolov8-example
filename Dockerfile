FROM  --platform=linux/amd64 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.13.1-gpu-py39

WORKDIR /opt/ml/code

COPY requirements.txt /opt/ml/code/
RUN pip3 install -r requirements.txt

COPY train.py /opt/ml/code/
COPY data.yaml /opt/ml/code/

ENV SAGEMAKER_PROGRAM train.py
