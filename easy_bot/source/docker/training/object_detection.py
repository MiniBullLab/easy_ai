#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

from easyai.utility.logger import EasyLogger
if EasyLogger.check_init():
    log_file_path = EasyLogger.get_log_file_path("ai_runtime.log")
    EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")

import os
import ast
import boto3
import time
import tarfile
from easy_tools.model_train.ai_train import EasyAiModelTrain


def train():
    data_bucket = os.environ.get("DATA_BUCKET")
    data_prefix = os.environ.get("DATA_PREFIX")
    model_bucket = os.environ.get("MODEL_BUCKET")

    print("DATA_BUCKET={}".format(data_bucket))
    print("DATA_PREFIX={}".format(data_prefix))
    print("MODEL_BUCKET={}".format(model_bucket))

    hyperparameters = ast.literal_eval(os.environ["HYPERPARAMETERS"])  # convert string expr to dict
    epochs = int(hyperparameters.get("EPOCHS", "80"))

    data_dir = '/opt/ml/dataset'
    output_dir = '/opt/ml/output'
    model_dir = '/opt/ml/model'

    for dirname in [data_dir, output_dir, model_dir]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # Download all objects
    print("Collecting objects in s3://{}/{}".format(data_bucket, data_prefix))
    s3 = boto3.client('s3', region_name=os.environ["AWS_DEFAULT_REGION"])
    all_keys = []
    task_list = []
    response = s3.list_objects_v2(Bucket=data_bucket, Prefix=data_prefix)
    while True:
        keys = response["Contents"]
        for ctx in keys:
            key = ctx["Key"]
            print(key)
            if key.endswith("/"):
                print(f"Skipping {data_bucket}/{key}.")
                continue
            relpath = key[len(data_prefix):].strip('/')
            dirname = os.path.dirname(relpath)
            basename = os.path.basename(relpath)
            fulldirname = os.path.join(data_dir, dirname)
            fullpath = os.path.join(fulldirname, basename)
            if not os.path.exists(fulldirname):
                os.makedirs(fulldirname)
            # s3.download_file(data_bucket, key, fullpath)
            task_list.append((data_bucket, key, fullpath))
            all_keys.append(fullpath)
        truncated = response["IsTruncated"]
        if not truncated:
            break
        token = response["NextContinuationToken"]
        response = s3.list_objects_v2(Bucket=data_bucket, Prefix=data_prefix, ContinuationToken=token)

    print("Downloading {} objects from s3://{}/{}/ to {}".format(len(all_keys), data_bucket, data_prefix, data_dir))

    tic = time.time()
    for data_bucket, key, fullpath in task_list:
        toc = time.time()
        if (toc - tic) > 10:
            print(f"Downloading s3://{data_bucket}/{key} to {fullpath}")
            tic = time.time()
        s3.download_file(data_bucket, key, fullpath)

    print("Downloaded {} objects.".format(len(all_keys)))
    print(os.listdir(data_dir))
    train_path = os.path.join(data_dir, "ImageSets/train.txt")
    val_path = os.path.join(data_dir, "ImageSets/val.txt")
    tools_dir = "/usr/local/lib/python3.6/dist-packages/easy_tools"
    train_process = EasyAiModelTrain(train_path, val_path, 0)
    train_process.det2d_model_train(tools_dir)

    # Create archive file for uploading to s3 bucket
    model_path = os.path.join(model_dir, "model.tar.gz")
    print("Creating compressed tar archive {}".format(model_path))

    tar = tarfile.open(model_path, "w:gz")
    for name in ["/opt/ml/code/.easy_log/snapshot/det2d_best.pt",
                 "/opt/ml/code/.easy_log/snapshot/denet.onnx",
                 "/opt/ml/code/.easy_log/config/detection2d_config.json",
                 "/opt/ml/code/.easy_log/det2d_evaluation.txt"]:
        basename = os.path.basename(name)
        tar.add(name, arcname=basename)
    tar.close()

    # Upload to S3 bucket
    model_prefix = os.environ.get("MODEL_PREFIX")
    print("Uploading tar archive {} to s3://{}/{}/".format(model_path, model_bucket, model_prefix))
    s3.upload_file(model_path, model_bucket, "{}/model.tar.gz".format(model_prefix))