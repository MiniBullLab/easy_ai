ARG REGISTRY_URI
FROM ${REGISTRY_URI}/mxnet-inference:1.6.0-gpu-py3
# FROM ${REGISTRY_URI}/mxnet-inference:1.8.0-gpu-py37-cu110-ubuntu16.04

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# install flask
RUN pip install -i https://opentuna.cn/pypi/web/simple/ flask gevent gunicorn opencv-python==3.4.11.43

# Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

# install flask
RUN pip install -i https://opentuna.cn/pypi/web/simple/ boto3

# Set up the program in the image
COPY serve.py /opt/program/
COPY predictor.py /opt/program/
COPY wsgi.py /opt/program/
COPY nginx.conf /opt/program/

WORKDIR /opt/program

ENTRYPOINT ["python", "serve.py"]
