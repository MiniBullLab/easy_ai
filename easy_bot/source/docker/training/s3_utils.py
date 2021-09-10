import boto3
import os
import asyncio
import functools

def list_objects(bucket, prefix):
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    all_keys = []
    while True:
        keys = response.get("Contents", None)
        if keys is None:
            break
        for ctx in keys:
            key = ctx["Key"]
            all_keys.append(key)
        truncated = response["IsTruncated"]
        if not truncated:
            break
        token = response["NextContinuationToken"]
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=token)
    return all_keys

def download_objects(bucket, keys, target_dir):
    pass
