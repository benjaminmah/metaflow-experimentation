#FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

RUN apt-get update -y \
    && apt-get -y install git

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt \
    && pip install --no-build-isolation flash-attn