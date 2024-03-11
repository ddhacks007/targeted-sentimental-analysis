FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
RUN pip3 install -r requirements.txt
COPY . .
COPY Main.py /opt/ml/code/train.py
RUN SAGEMAKER_PROGRAM Main.py
