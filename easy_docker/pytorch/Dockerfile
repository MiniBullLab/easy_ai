ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && \
    apt-get install -y software-properties-common libsm6 libxext6 git \
    libglib2.0-0 libsm6 libxrender-dev libxext6 && \
    apt-get clean

COPY pytorch/add_user.sh pytorch/requirements.txt /scripts/

RUN pip install -i https://mirrors.aliyun.com/pypi/simple -r /scripts/requirements.txt && \
    echo "export PYTHONPATH=/easy_data/easy_ai:\$PYTHONPATH">>/etc/bash.bashrc && \
    /bin/bash -c "source /etc/bash.bashrc"

RUN mkdir /easy_data
WORKDIR /easy_data
