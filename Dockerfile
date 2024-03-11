FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
COPY . .
RUN pip3 install sagemaker-training
RUN pip3 install -r requirements.txt
COPY . /opt/ml/code/
ENV SAGEMAKER_PROGRAM Main.py