ARG REGISTRY_URI
FROM ${REGISTRY_URI}/mxnet-inference:1.6.0-gpu-py3

RUN pip install -i https://opentuna.cn/pypi/web/simple autogluon==0.0.12

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

RUN apt update
RUN apt install -y unzip

RUN mkdir -p /root/.mxnet/models/

# image classification pre-trained model
RUN wget -P /root/.mxnet/models/ https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/resnet50_v2-ecdde353.zip
RUN cd /root/.mxnet/models/ && unzip resnet50_v2-ecdde353.zip
RUN rm -f /root/.mxnet/models/resnet50_v2-ecdde353.zip

# object detection pre-trained model
RUN wget -P /root/.mxnet/models/ https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/yolo3_darknet53_coco-09767802.zip
RUN cd /root/.mxnet/models/ && unzip yolo3_darknet53_coco-09767802.zip
RUN rm -f /root/.mxnet/models/yolo3_darknet53_coco-09767802.zip

# named entity recognition pre-trained model
RUN wget -P /root/.mxnet/models/ https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/vocab/wiki_cn_cased-ddebd8f3.zip
RUN wget -P /root/.mxnet/models/ https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/bert_12_768_12_wiki_cn_cased-885ebb9a.zip
RUN cd /root/.mxnet/models/ && unzip wiki_cn_cased-ddebd8f3.zip
RUN cd /root/.mxnet/models/ && unzip bert_12_768_12_wiki_cn_cased-885ebb9a.zip
RUN rm -f /root/.mxnet/models/wiki_cn_cased-ddebd8f3.zip
RUN rm -f /root/.mxnet/models/bert_12_768_12_wiki_cn_cased-885ebb9a.zip

RUN python -m pip install -i https://opentuna.cn/pypi/web/simple boto3
RUN python -m pip install -i https://opentuna.cn/pypi/web/simple opencv-python==3.2.0.8

COPY train.py /opt/ml/code/
COPY image_classification.py /opt/ml/code/
COPY object_detection.py /opt/ml/code/
COPY named_entity_recognition.py /opt/ml/code/
COPY ner_utils.py /opt/ml/code/
COPY voc_utils.py /opt/ml/code/
COPY s3_utils.py /opt/ml/code/
WORKDIR /opt/ml/code

ENTRYPOINT ["python", "-u", "train.py"]

