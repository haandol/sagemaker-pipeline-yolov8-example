# Sagemaker Custom Pytorch Docker Yolov8

# Prerequisites

- Docker
- AWS CLI
- Jupyter Notebook (for testing)

# Using dataset

[Pikachu Detection](https://universe.roboflow.com/oklahoma-state-university-jyn38/pikachu-detection) by Roboflow

# Build and push training image on your ECR

run `./build_and_push.sh` on your terminal

# Create sagemaker studio

https://github.com/haandol/cdk-sagemaker-studio

# Create sagemaker pipeline for training model

Run [pikachu_sagemaker.ipynb](/notebook/pikachu_sagemaker.ipynb)

# Test

Run [pikachu_sagemaker.ipynb](/notebook/pikachu_sagemaker.ipynb)
