#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""4. Transfer Learning with Your Own Image Dataset
=======================================================

Dataset size is a big factor in the performance of deep learning models.
``ImageNet`` has over one million labeled images, but
we often don't have so much labeled data in other domains.
Training a deep learning models on small datasets may lead to severe overfitting.

Transfer learning is a technique that addresses this problem.
The idea is simple: we can start training with a pre-trained model,
instead of starting from scratch.
As Isaac Newton said, "If I have seen further it is by standing on the
shoulders of Giants".
"""
import os
import boto3
from datetime import datetime
import numpy as np
import tarfile

task_type = os.environ.get("TASK_TYPE")
task_name = os.environ.get("TASK_NAME")
model_bucket = os.environ.get("MODEL_BUCKET")

print("TASK_TYPE={}".format(task_type))

if task_type == "IMAGE_CLASSIFICATION":
    from image_classification import train
elif task_type == "OBJECT_DETECTION":
    from object_detection import train
elif task_type == "NAMED_ENTITY_RECOGNITION":
    from image_segmentattion import train

train()


def init_efs():
    print("Downloading to EFS...", flush=True)

    s3 = boto3.client('s3', region_name=os.environ["AWS_DEFAULT_REGION"])

    efs_path = "/mnt/ml"
    task_type_map = {
        "IMAGE_CLASSIFICATION": "Image Classification",
        "OBJECT_DETECTION": "Object Detection",
        "NAMED_ENTITY_RECOGNITION": "Named Entity Recognition",
        "": "Image Classification",
    }
    task_type_map_s3 = {k:v.lower().replace(" ", "-") for k, v in task_type_map.items()}
    task_type_s3 = task_type_map_s3[task_type]

    def get_datetime_str(dt=datetime.now()):
        tt = dt.timetuple()
        prefix = tt[0]
        name = '-'.join(['{:02}'.format(t) for t in tt[1:-3]])
        suffix = '{:03d}'.format(dt.microsecond)[:3]
        job_name_suffix = "{}-{}-{}".format(prefix, name, suffix)
        return job_name_suffix

    model_dir = os.path.join(efs_path, 'model')
    model_prefix = "{}/{}".format(task_type_s3, task_name)
    key = "{}/model.tar.gz".format(model_prefix)

    response = s3.get_object(Bucket=model_bucket, Key=key)
    LastModified = response.get("LastModified")
    dt_str = get_datetime_str(LastModified)
    target_dir = os.path.join(model_dir, task_type_s3, f"{task_name}-{dt_str}")

    if not os.path.exists(os.path.join(target_dir, "DONE")):
        os.makedirs(target_dir, exist_ok=True)
        fullpath = os.path.join(target_dir, "model.tar.gz")

        if not os.path.exists(os.path.join(target_dir, "DOWNLOADED")):
            print("Downloading from s3://{}/{}/model.tar.gz to {}".format(model_bucket, model_prefix, target_dir))
            s3.download_file(model_bucket, key, fullpath)
            np.savetxt(os.path.join(target_dir, "DOWNLOADED"), [])

        if not os.path.exists(os.path.join(target_dir, "EXTRACTED")):
            print("Extracting {} to {}".format(fullpath, target_dir))
            tar = tarfile.open(fullpath)
            tar.extractall(path=target_dir)
            tar.close()
            np.savetxt(os.path.join(target_dir, "EXTRACTED"), [])

        np.savetxt(os.path.join(target_dir, "DONE"), [])


init_efs()

print("Done.", flush=True)
