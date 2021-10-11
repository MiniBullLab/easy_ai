import boto3
import os
import asyncio
import functools
import time
from iam_helper import IamHelper

def get_object_url(bucket, key):
    s3 = boto3.client('s3')
    url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={
            'Bucket': bucket,
            'Key': key
        }
    )
    return url


def put_object(bucket, key, img_bytes):
    s3 = boto3.client("s3")
    response = s3.put_object(
        Body = img_bytes,
        Bucket = bucket,
        Key = key
    )
    return get_object_url(bucket, key)


def list_objects(bucket, prefix):
    s3 = boto3.client("s3")
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


def recursive_copy(src_bucket, src_prefix, dst_bucket, dst_prefix):
    tic = time.time()
    def get_partition():
        if boto3.session.Session().region_name in ["cn-northwest-1", "cn-north-1"]:
            return "aws-cn"
        else:
            return "aws"
    partition = get_partition()
    region_name = boto3.session.Session().region_name
    s3 = boto3.client("s3", region_name=region_name)
    s3control = boto3.client('s3control', region_name=region_name)
    account_id = boto3.client('sts').get_caller_identity().get('Account')

    if src_prefix[-1] == "/":
        src_prefix = src_prefix[:-1]
    if dst_prefix[-1] == "/":
        dst_prefix = dst_prefix[:-1]

    # batchOperationsRole = f"arn:{partition}:iam::{account_id}:role/AdminRoleForS3BatchOperations"
    batchOperationsRole = os.environ["BATCH_OPS_ROLE_ARN"]

    # collect all keys in the bucket and save to manifest.csv file
    all_keys = list_objects(src_bucket, src_prefix)
    manifest_csv = ""
    for key in all_keys:
        manifest_csv += f"{src_bucket},{key}\n"
    
    # upload the manifest file to s3://{src_bucket}/{src_prefix}/manifest.csv
    manifest_key = f"{src_prefix}/manifest.csv"
    manifest_uri = f"s3://{src_bucket}/{manifest_key}"
    manifest_arn = f"arn:{partition}:s3:::{src_bucket}/{manifest_key}"
    response = s3.put_object(
        Body = manifest_csv,
        Bucket = src_bucket,
        Key = manifest_key
    )
    etag = boto3.resource('s3').Object(src_bucket, manifest_key).e_tag.strip('"')
    print(f"Uploaded manifest to {manifest_uri}.")

    dst_bucket_arn = f"arn:{partition}:s3:::{dst_bucket}"

    kwargs = {
        "AccountId": account_id,
        "ConfirmationRequired": False,
        "RoleArn": batchOperationsRole,
        "Priority": 10,
        "Manifest": {
            "Spec": {
                "Format": "S3BatchOperations_CSV_20180820",
                "Fields": ["Bucket", "Key"],
            },
            "Location": {
                "ObjectArn": manifest_arn,
                "ETag": etag
            },
        },
        "Operation": {
            'S3PutObjectCopy': {
                "TargetResource": dst_bucket_arn,
                'TargetKeyPrefix': dst_prefix,
                "MetadataDirective": "COPY",
                "RequesterPays": False,
                "StorageClass": "STANDARD",
            },
        },
        "Report": {
            'Enabled': False,
        },
    }

    import json
    print(json.dumps(kwargs, indent=4))

    s3control.create_job(**kwargs)

    # wait for the s3 batch operations job
    while True:
        toc = time.time()
        if (toc - tic) > 18:
            break
        time.sleep(1)

    return len(all_keys)

# async def main():
#     pass

def recursive_copy_inplace(src_bucket, src_prefix, dst_bucket, dst_prefix):
    def get_partition():
        if boto3.session.Session().region_name in ["cn-northwest-1", "cn-north-1"]:
            return "aws-cn"
        else:
            return "aws"
    partition = get_partition()
    region_name = boto3.session.Session().region_name
    s3 = boto3.client("s3", region_name=region_name)
    s3control = boto3.client('s3control', region_name=region_name)
    account_id = boto3.client('sts').get_caller_identity().get('Account')

    if src_prefix[-1] == "/":
        src_prefix = src_prefix[:-1]
    if dst_prefix[-1] == "/":
        dst_prefix = dst_prefix[:-1]

    # collect all keys in the bucket
    response = s3.list_objects_v2(Bucket=src_bucket, Prefix=src_prefix)
    all_keys = []
    while True:
        keys = response["Contents"]
        for ctx in keys:
            src_key = ctx["Key"]
            all_keys.append(src_key)
        truncated = response["IsTruncated"]
        if not truncated:
            break
        token = response["NextContinuationToken"]
        response = s3.list_objects_v2(Bucket=src_bucket, Prefix=src_prefix, ContinuationToken=token)

    keypairs = []
    for src_key in all_keys:
        basename = os.path.basename(src_key)
        dst_key = f"{dst_prefix}/{basename}"
        keypairs.append((src_key, dst_key))

    os.environ["SRC_BUCKET"] = src_bucket
    os.environ["SRC_PREFIX"] = src_prefix
    os.environ["DST_BUCKET"] = dst_bucket
    os.environ["DST_PREFIX"] = dst_prefix
    print(f"src_bucket={src_bucket}, src_prefix={src_prefix}, dst_bucket={dst_bucket}, dst_prefix={dst_prefix}")
    print(f"key_count={len(all_keys)}")
    
    async def main():
        loop = asyncio.get_running_loop()
        src_bucket = os.environ["SRC_BUCKET"]
        src_prefix = os.environ["SRC_PREFIX"]
        dst_bucket = os.environ["DST_BUCKET"]
        dst_prefix = os.environ["DST_PREFIX"]
        objects = await asyncio.gather(
            *[
                loop.run_in_executor(None, functools.partial(s3.copy_object, Bucket=dst_bucket, Key=dst_key, CopySource={'Bucket': src_bucket, 'Key': src_key}))
                for src_key, dst_key in keypairs
            ]
        )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

    return len(all_keys)
