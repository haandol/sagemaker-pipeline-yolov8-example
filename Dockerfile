FROM  --platform=linux/amd64 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.13.1-gpu-py39

RUN pip3 install ultralytics

COPY train.py /opt/ml/code/train.py

ENV SAGEMAKER_PROGRAM train.py