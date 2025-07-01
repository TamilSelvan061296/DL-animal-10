FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

RUN pip install uv

WORKDIR /app