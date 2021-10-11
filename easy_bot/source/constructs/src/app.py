import json
import os
import sys
import time

from sagemaker_controller import create_endpoint
from ecs_controller import create_training_job, run_task, \
    describe_training_job, describe_endpoint, \
    invoke_endpoint, list_tasks, stop_task, delete_task, describe_task_runtime, \
    create_auto_scaling_group, create_training_job_v2, invoke_endpoint_v2, create_endpoint_v2
from dynamodb_controller import TaskTable
from predictor import invoke_endpoint_serverless

from s3_controller import get_object_url, put_object, recursive_copy, recursive_copy_inplace, list_objects
from boto3.session import Session
from utils import get_datetime_str, get_utctime, format_datetime, AuthPolicy
import boto3
import base64
import traceback
import asyncio
import ast
from datetime import datetime, timezone
from model import task_type_map, task_type_map_s3, task_type_lut
from urllib.parse import unquote

def get_headers():
    return {
        "Content-Type": "application/json",
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET,DELETE',
    }


def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

    context: object, required
        Lambda Context runtime methods and attributes

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict
    """
    tic = time.time()

    s3 = boto3.client("s3")
    taskTable = TaskTable()
    dataBucket = os.environ["DATA_BUCKET"]

    toc = time.time()
    print(f"app -0- elapsed: {(toc - tic)*1000.0} ms")

    resource = event.get("source", None)
    if resource == "aws.sagemaker":
        detail = event["detail"]
        job_id = detail["TrainingJobName"]
        transitions = detail["SecondaryStatusTransitions"]
        transition0 = transitions[-1]
        status = transition0["Status"]
        status_msg = transition0["StatusMessage"]
    elif resource == "aws.ecs":
        detail = event["detail"]
        container0 = detail["containers"][0]
        status = container0["lastStatus"]
        taskArn = container0["taskArn"]
        job_id = taskArn.split("/")[-1]

    elif resource == "aws.autoscaling":
        detail = event["detail"]
        status = detail["Description"].split(" ")[0]
        asg_name = detail["AutoScalingGroupName"]
        family = ""
        job_id = ""
        if asg_name.startswith("MLBot-Training"):
            job_id = asg_name[len("MLBot-Training-"):]
            family = "Training"
        elif asg_name.startswith("MLBot-Inference"):
            job_id = asg_name[len("MLBot-Inference-"):]
            family = "Inference"
        else:
            print(f"Unexpected AutoScaling group {asg_name}")

    elif "httpMethod" in event:
        httpMethod = event["httpMethod"]
        resource = event["resource"]
        body_str = event["body"]
    elif "authorizationToken" in event:
        resource = "aws.apigw"
    else:
        resource = "aws.s3control"

    response = {}
    if resource == "/hello":
        response = {
            "statusCode": 200,
            "body": html,
            "headers": {
                "Content-Type": "text/html",
            },
        }

    elif resource == "/upload":
        print(body_str)
        body_dict = json.loads(body_str)
        print(body_dict)
        job_id = body_dict["job"]

        # query task type from task table
        task_type = taskTable.get_item(job_id, "TaskType")
        if task_type is None:
            task_type = "IMAGE_CLASSIFICATION"
        task_type_s3 = task_type_map_s3[task_type]

        class_id = body_dict["class"]
        filename = body_dict["filename"]
        img_base64 = body_dict["data"]
        print(img_base64)
        img_base64 = img_base64.split(",")[1]
        print(img_base64)
        img_bytes = base64.decodebytes(img_base64.encode("utf-8"))

        bucket = os.environ["DATA_BUCKET"]
        key = "{}/{}/{}/{}".format(task_type_s3, job_id, class_id, filename)
        obj_url = put_object(bucket, key, img_bytes)

        if task_type == "IMAGE_CLASSIFICATION":
            classes_str = taskTable.get_item(job_id, "Classes")
            if classes_str is None or classes_str == "":
                classes_str = "[]"
            classes = json.loads(classes_str)
            if class_id not in classes:
                classes.append(class_id)
            classes_str = json.dumps(classes)
            taskTable.update_item(job_id, "Classes", classes_str)

        body_str = json.dumps({"Status": "Success", "Bucket": bucket, "Key": key,
                               "ObjectUrl": obj_url})

        response = {
            "statusCode": 200,
            "body": body_str,
            "headers": get_headers(),
        }

    elif resource == "/status":
        body_dict = json.loads(body_str)
        job_id = body_dict.get("endpoint", "")
        print("STATUS: endpoint={}".format(job_id))

        msg = {}
        try:
            transitions, artifacts, cloudwatch_logs_url = describe_training_job(job_id)
            status, name = describe_endpoint(job_id)
            msg = {"Status": "Success", "TrainingJobStatus": transitions, \
                   "EndpointStatus": status, "EndpointName": job_id, \
                   "ModelArtifacts": artifacts}
        except Exception as e:
            msg = {"Status": "Failed", "Message": str(e)}
            traceback.print_exc()
        print(msg)

        response = {
            "statusCode": 200,
            "body": json.dumps(msg),
            "headers": get_headers(),
        }

    elif resource.startswith("/tasks"):
        print(body_str)
        if resource == "/tasks":
            if httpMethod == "GET":
                # list task properties that is stored in dynamodb tables
                taskNames, taskTypes, taskRuntimes, statusNames, createTimes, updateTimes, nextToken = list_tasks()

                # map task type definition (e.g. IMAGE_CLASSIFICATION) to task type name (e.g. Image Classification)
                # taskTypes = [task_type_map[task_type] for task_type in taskTypes]
                createTimes = [format_datetime(createTime) for createTime in createTimes]
                updateTimes = [format_datetime(updateTime) for updateTime in updateTimes]

                tasks = []
                for name, taskType, elapsed, status, createTime, updateTime in \
                    zip(taskNames, taskTypes, taskRuntimes, statusNames, createTimes, updateTimes):
                    tasks.append({"taskName": name,
                                  "taskType": taskType,
                                  "taskStatus": status,
                                  "taskRunningTime": elapsed,
                                  "createTime": createTime,
                                  "updateTime": updateTime})
                body = {"tasks": tasks}
                if nextToken is not None:
                    body.update({"nextToken": nextToken})
                body_str = json.dumps(body)

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

            elif httpMethod == "POST":
                body_dict = json.loads(body_str)
                job_id = body_dict["taskId"]
                task_type_def = body_dict["taskType"]
                status, msg = "Success", ""
                if taskTable.get_item(job_id) is None:
                    taskTable.put_item(job_id, task_type_def)
                    taskTable.update_item(job_id, "CreatedAt", get_utctime())
                    taskTable.update_item(job_id, "UpdatedAt", get_utctime())
                    mg = "Created task {} with task type {}".format(job_id, task_type_def)
                else:
                    status = "Failed"
                    msg = "Task name already exists"
                body = {"Status": status, "Message": msg}
                body_str = json.dumps(body)

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

        elif resource == "/tasks/{task_id}":
            pathParameters = event["pathParameters"]
            job_id = unquote(pathParameters["task_id"])

            if httpMethod == "DELETE":
                try:
                    status, msg = delete_task(job_id)
                except Exception as e:
                    status = "Failed"
                    msg = str(e)
                    traceback.print_exc()
                body = {"Status": status, "taskName": job_id, "Message": msg}
                body_str = json.dumps(body)

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

            elif httpMethod == "GET":
                msg = {}
                try:
                    transitions, artifacts, cloudwatch_logs_url = describe_training_job(job_id)
                    status, name = describe_endpoint(job_id)
                    task_type = taskTable.get_item(job_id, "TaskType")
                    taskData = {}
                    if task_type == "IMAGE_CLASSIFICATION":
                        taskData = {
                            "S3URIs": ["s3://{}/{}/{}/".format(dataBucket, task_type_map_s3[task_type], job_id)],
                        }
                    elif task_type == "OBJECT_DETECTION":
                        taskData = {
                            "S3URIs": ["s3://{}/{}/{}/".format(dataBucket, task_type_map_s3[task_type], job_id)],
                        }
                    elif task_type == "NAMED_ENTITY_RECOGNITION":
                        taskData = {
                            "S3URIs": ["s3://{}/{}/{}/".format(dataBucket, task_type_map_s3[task_type], job_id)],
                        }

                    msg = {"Status": "Success",
                           "TaskType": task_type,
                           "TaskName": job_id,
                           "TaskDataSource": "S3",
                           "TaskData": taskData,
                           "TrainingJobStatus": transitions, \
                           "EndpointStatus": status, "EndpointName": job_id, \
                           "ModelArtifacts": artifacts}
                except Exception as e:
                    msg = {"Status": "Failed", "Message": str(e)}
                    traceback.print_exc()
                print(msg)
                body_str = json.dumps(msg)

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

        elif resource == "/tasks/{task_id}/status":
            pathParameters = event["pathParameters"]
            job_id = unquote(pathParameters["task_id"])

            if httpMethod == "GET":
                msg = {}
                try:
                    transitions, artifacts, cloudwatch_logs_url = describe_training_job(job_id)
                    status, name = describe_endpoint(job_id)
                    task_type = taskTable.get_item(job_id, "TaskType")
                    if task_type is None:
                        raise RuntimeError(f"Failed to get task type of the '{job_id}' task")
                    task_type_s3 = task_type_map_s3[task_type]
                    hyperparameters_str = taskTable.get_item(job_id, "HyperParameters")
                    if hyperparameters_str is None:
                        hyperparameters_str = "{}"
                    hyperparameters = ast.literal_eval(hyperparameters_str)
                    taskData = {}
                    taskData = {
                        "S3URIs": ["s3://{}/{}/{}/".format(dataBucket, task_type_s3, job_id)],
                    }

                    msg = {
                        "Status": "Success",
                        "TaskType": task_type,
                        "TaskName": job_id,
                        "TaskDataSource": "S3",
                        "TaskData": taskData,
                        "TrainingJobStatus": transitions,
                        "EndpointStatus": status, "EndpointName": job_id,
                        "ModelArtifacts": artifacts,
                        "CloudWatchLogsUrl": cloudwatch_logs_url,
                        "HyperParameters": hyperparameters
                    }
                except Exception as e:
                    msg = {"Status": "Failed", "Message": str(e)}
                    traceback.print_exc()

                print(msg)
                body_str = json.dumps(msg)

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

        elif resource == "/tasks/{task_id}/stop":
            pathParameters = event["pathParameters"]
            job_id = unquote(pathParameters["task_id"])
            try:
                status, msg = stop_task(job_id)
            except Exception as e:
                status = "Failed"
                msg = str(e)
                traceback.print_exc()
            body = {"Status": status, "taskName": job_id, "Message": msg}
            body_str = json.dumps(body)

            response = {
                "statusCode": 200,
                "body": body_str,
                "headers": get_headers(),
            }

        elif resource == "/tasks/{task_id}/data":

            if httpMethod == "GET":
                pathParameters = event["pathParameters"]
                job_id = unquote(pathParameters["task_id"])
                keycounts = []
                obj_uri_l = []

                try:
                    task_type = taskTable.get_item(job_id, "TaskType")
                    task_type_s3 = task_type_map_s3[task_type]

                    if task_type == "IMAGE_CLASSIFICATION":
                        classes_str = taskTable.get_item(job_id, "Classes")
                        if classes_str is None or classes_str == "":
                            classes_str = "[]"
                        classes = json.loads(classes_str)

                        obj_uri_str = taskTable.get_item(job_id, "OriginURIs")
                        obj_uri_str = "{}" if obj_uri_str is None else obj_uri_str
                        obj_uri_map = ast.literal_eval(obj_uri_str)

                        for class_id in classes:
                            prefix = f"{task_type_s3}/{job_id}/{class_id}/"
                            objs = list_objects(dataBucket, prefix)
                            keycounts.append(len(objs))
                            if class_id in obj_uri_map:
                                obj_uri_l.append(obj_uri_map[class_id])
                            else:
                                obj_uri_l.append("")
                    else:
                        classes = []
                        keycounts = []
                        obj_uri_l = []

                    task_status = taskTable.get_item(job_id, "Status")
                    status, msg = "Success", ""

                except Exception as e:
                    status = "Failed"
                    msg = str(e)
                    traceback.print_exc()

                body = {
                    "Status": status, "Message": msg, 
                    "taskName": job_id, "taskType": task_type, "taskStatus": task_status,
                    "taskDataSource": "S3",
                    "taskData": {
                        "Classes": classes,
                        "SampleCount": keycounts,
                        "OriginURIs": obj_uri_l,
                    },
                }
                body_str = json.dumps(body)

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

        elif resource == "/tasks/{task_id}/data/{class_id}":
            pathParameters = event["pathParameters"]
            queryStringParameters = event["queryStringParameters"]
            job_id = unquote(pathParameters["task_id"])
            class_id = unquote(pathParameters["class_id"])

            if httpMethod == "POST":
                # query task type from task table
                task_type = taskTable.get_item(job_id, "TaskType")
                task_type_s3 = task_type_map_s3[task_type]

                try:
                    status, msg = "Success", ""
                    body_dict = json.loads(body_str)
                    filename = body_dict["filename"]
                    img_base64 = body_dict["data"]
                    img_base64 = img_base64.split(",")[1]
                    img_bytes = base64.decodebytes(img_base64.encode("utf-8"))
                    bucket = os.environ["DATA_BUCKET"]
                    key = "{}/{}/{}/{}".format(task_type_s3, job_id, class_id, filename)
                    obj_url = put_object(bucket, key, img_bytes)

                    # update classes list in task table
                    if task_type == "IMAGE_CLASSIFICATION":
                        classes_str = taskTable.get_item(job_id, "Classes")
                        if classes_str is None or classes_str == "":
                            classes_str = "[]"
                        classes = json.loads(classes_str)
                        if class_id not in classes:
                            classes.append(class_id)
                        classes_str = json.dumps(classes)
                        taskTable.update_item(job_id, "Classes", classes_str)

                except Exception as e:
                    status = "Failed"
                    msg = str(e)
                    traceback.print_exc()

                body = {"Status": status, "Message": msg,
                        "taskName": job_id, "className": class_id,
                        "Bucket": bucket, "Key": key, "ObjectUrl": obj_url}
                body_str = json.dumps(body)

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

            elif httpMethod == "GET":
                # query task type from task table
                task_type = taskTable.get_item(job_id, "TaskType")
                task_type_s3 = task_type_map_s3[task_type]
                keycount = 0

                try:
                    status, msg, keys = "Success", "", []
                    IsTruncated = False
                    NextContinuationToken = ""

                    ContinuationToken = None
                    MaxKeys = 4

                    if queryStringParameters:
                        if "ContinuationToken" in queryStringParameters:
                            ContinuationToken = queryStringParameters["ContinuationToken"]
                        if "MaxKeys" in queryStringParameters:
                            MaxKeys = int(queryStringParameters["MaxKeys"])

                    if task_type in ["IMAGE_CLASSIFICATION"]:
                        prefix = f"{task_type_s3}/{job_id}/{class_id}/"
                        if ContinuationToken:
                            response = s3.list_objects_v2(Bucket=dataBucket, Prefix=prefix, MaxKeys=MaxKeys,
                                                          ContinuationToken=ContinuationToken)
                        else:
                            response = s3.list_objects_v2(Bucket=dataBucket, Prefix=prefix, MaxKeys=MaxKeys)
                        IsTruncated = response["IsTruncated"]
                        if IsTruncated:
                            NextContinuationToken = response["NextContinuationToken"]
                        urls = []
                        if "Contents" in response:
                            keys = [content["Key"] for content in response["Contents"]]
                            urls = [get_object_url(dataBucket, key) for key in keys]

                        objs = list_objects(dataBucket, prefix)
                        keycount = len(objs)

                except Exception as e:
                    status = "Failed"
                    msg = str(e)
                    traceback.print_exc()

                body = {
                    "Status": status, "Message": msg,
                    "taskName": job_id, "className": class_id,
                    "IsTruncated": IsTruncated, "NextContinuationToken": NextContinuationToken,
                    "URLs": urls, "SampleCount": keycount
                }
                body_str = json.dumps(body)

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

        elif resource == "/tasks/{task_id}/s3data":
            pathParameters = event["pathParameters"]
            job_id = unquote(pathParameters["task_id"])

            if httpMethod == "GET":
                status, msg, keys, obj_uri = "Success", "", [], []
                try:
                    task_type = taskTable.get_item(job_id, "TaskType")
                    task_type_s3 = task_type_map_s3[task_type]
                    obj_uri_str = taskTable.get_item(job_id, "OriginURIs")
                    obj_uri_str = "{}" if obj_uri_str is None else obj_uri_str
                    obj_uri = ast.literal_eval(obj_uri_str)

                    response = s3.list_objects_v2(Bucket=dataBucket, Prefix=task_type_s3 + '/', Delimiter = "/")
                    keys = [content["Prefix"] for content in response["CommonPrefixes"]]
                except Exception as e:
                    status = "Failed"
                    msg = str(e)
                    traceback.print_exc()
                body = {"Status": status, "taskName": job_id, "Message": msg, "Keys": keys, "OriginURIs": obj_uri}
                body_str = json.dumps(body)

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

            elif httpMethod == "POST":
                try:
                    status, msg = "Success", ""
                    body_dict = json.loads(body_str)
                    obj_uri = body_dict["OriginURIs"]
                    obj_uri_str = str(obj_uri)
                    task_type = taskTable.get_item(job_id, "TaskType")
                    task_type_s3 = task_type_map_s3[task_type]

                    taskTable.update_item(job_id, "DataSource", "S3")

                    if not isinstance(obj_uri, list):
                        msg = "OriginURIs input should be a list, but got {}".format(type(obj_uri))
                        raise RuntimeError(msg)

                    if task_type == "IMAGE_CLASSIFICATION":
                        if len(obj_uri) > 0:
                            class_id = body_dict.get("ClassId", None)
                            origin_uri = obj_uri[0]
                            if class_id is None:
                                msg = "missing input argument 'ClassId' in RESTful api."
                                raise RuntimeError(msg)
                            if len(obj_uri[0]) < 5:
                                raise RuntimeError("Invalid origin URI.")
                            from urllib.parse import urlparse
                            url = urlparse(obj_uri[0], allow_fragments=False)
                            src_bucket = url.netloc
                            src_prefix = url.path[1:]
                            dst_prefix = f"{task_type_s3}/{job_id}/{class_id}"

                            # update classes list in task table
                            classes_str = taskTable.get_item(job_id, "Classes")
                            obj_uri_str = taskTable.get_item(job_id, "OriginURIs")
                            if classes_str is None or classes_str == "":
                                classes_str = "[]"
                            classes = json.loads(classes_str)
                            obj_uri_str = "{}" if obj_uri_str is None else obj_uri_str
                            obj_uri_map = ast.literal_eval(obj_uri_str)
                            if class_id not in classes:
                                classes.append(class_id)
                            obj_uri_map[class_id] = origin_uri
                            obj_uri_l = []
                            for cls in classes:
                                if cls in obj_uri_map:
                                    obj_uri_l.append(obj_uri_map[cls])
                                else:
                                    obj_uri_l.append("")
                            classes_str = json.dumps(classes)
                            obj_uri_map_str = json.dumps(obj_uri_map)

                            taskTable.update_item(job_id, "Classes", classes_str)
                            taskTable.update_item(job_id, "OriginURIs", obj_uri_map_str)

                            key_count = recursive_copy_inplace(src_bucket, src_prefix, dataBucket, dst_prefix)
                            msg = f"Copied {key_count} objects from {obj_uri[0]} to s3://{dataBucket}/{dst_prefix} for {class_id} class."

                        else:
                            msg = "No object listed."
                            raise RuntimeError(msg)

                    elif task_type == "OBJECT_DETECTION":
                        if len(obj_uri) > 0:
                            if len(obj_uri[0]) < 5:
                                raise RuntimeError("Invalid origin URI.")
                            from urllib.parse import urlparse
                            url = urlparse(obj_uri[0], allow_fragments=False)
                            src_bucket = url.netloc
                            src_prefix = url.path[1:]
                            dst_prefix = f"{task_type_s3}/{job_id}"

                            key_count = recursive_copy(src_bucket, src_prefix, dataBucket, dst_prefix)
                            taskTable.update_item(job_id, "DataPrefix", f"{dst_prefix}/{src_prefix}")
                            taskTable.update_item(job_id, "OriginURIs", obj_uri_str)

                            msg = f"Copied {key_count} objects from {obj_uri[0]} to s3://{dataBucket}/{dst_prefix}."
                        else:
                            msg = "No object listed."
                            raise RuntimeError(msg)

                    elif task_type == "NAMED_ENTITY_RECOGNITION":
                        if len(obj_uri) > 0:

                            taskTable.update_item(job_id, "OriginURIs", obj_uri_str)

                            key = f"{task_type_s3}/{job_id}/train.txt"
                            from urllib.parse import urlparse
                            url = urlparse(obj_uri[0], allow_fragments=False)
                            src_bucket = url.netloc
                            src_key = url.path[1:]
                            msg = f"Copied object from s3://{src_bucket}/{src_key} to s3://{dataBucket}/{key}"
                            print(msg)
                            s3.copy_object(Bucket=dataBucket, Key=key, CopySource={
                                "Bucket": src_bucket, "Key": src_key
                            })
                        else:
                            msg = "No object listed."
                            raise RuntimeError(msg)
                    else:
                        msg = f"Failed to import data for task type {task_type}."
                        raise RuntimeError(msg)
                except Exception as e:
                    status = "Failed"
                    msg = str(e)
                    traceback.print_exc()
                body = {"Status": status, "taskName": job_id, "Message": msg}
                body_str = json.dumps(body)

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

        elif resource == "/tasks/{task_id}/train_v2":
            if httpMethod == "POST":
                pathParameters = event["pathParameters"]
                job_id = unquote(pathParameters["task_id"])
                task_type = taskTable.get_item(job_id, "TaskType")

                bucket = os.environ["DATA_BUCKET"]
                msg = {}
                try:
                    hyperparameters = {}
                    if body_str and body_str != "":
                        body_dict = json.loads(body_str)
                        hyperparameters = body_dict.get("HyperParameters", {})
                        taskTable.update_item(job_id, "HyperParameters", str(hyperparameters))

                    taskId = create_training_job(bucket, job_id, task_type, hyperparameters)

                    msg = {"Status": "Success", "TrainingJob": job_id}
                except Exception as e:
                    msg = {"Status": "Failed", "Message": str(e)}
                    traceback.print_exc()
                print(msg)
                body_str = json.dumps(msg)

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

        # elif resource == "/tasks/{task_id}/train_v2":
        elif resource == "/tasks/{task_id}/train":
            if httpMethod == "POST":
                pathParameters = event["pathParameters"]
                job_id = unquote(pathParameters["task_id"])
                task_type = taskTable.get_item(job_id, "TaskType")

                bucket = os.environ["DATA_BUCKET"]
                msg = {}
                try:
                    instanceId = create_auto_scaling_group(job_id, "Training")

                    hyperparameters = {}
                    if body_str and body_str != "":
                        body_dict = json.loads(body_str)
                        hyperparameters = body_dict.get("HyperParameters", {})
                        taskTable.update_item(job_id, "HyperParameters", str(hyperparameters))
                    if instanceId != "":
                        # If instanceId == "", we would rely on EventBridge to
                        # capture instance status
                        taskId = create_training_job_v2(bucket, job_id, task_type,
                                                        hyperparameters, instanceId)

                    taskTable.update_item(job_id, "TrainingTaskId", "")
                    taskTable.update_item(job_id, "InferenceTaskId", "")
                    taskTable.update_item(job_id, "TrainingDuration", "0:00:00")
                    taskTable.update_item(job_id, "Status", "Training")
                    taskTable.update_item(job_id, "UpdatedAt", get_utctime())

                    msg = {"Status": "Success", "TrainingJob": job_id}
                except Exception as e:
                    msg = {"Status": "Failed", "Message": str(e)}
                    traceback.print_exc()
                print(msg)
                body_str = json.dumps(msg)

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

        elif resource == "/tasks/{task_id}/predict_v0":

            if httpMethod == "POST":
                pathParameters = event["pathParameters"]
                job_id = unquote(pathParameters["task_id"])
                outputs_str = "{}"

                try:
                    status, msg = "Success", ""
                    body_dict = json.loads(body_str)
                    task_type = taskTable.get_item(job_id, "TaskType")

                    toc = time.time()
                    print(f"app -1- elapsed: {(toc - tic)*1000.0} ms")

                    if task_type in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION"]:
                        img_base64 = body_dict["imagedata"]
                        img_base64_split = img_base64.split(",")
                        if len(img_base64_split) != 2:
                            raise RuntimeError("invalid imagedata")
                        img_base64 = img_base64_split[1]
                        img = base64.decodebytes(img_base64.encode("utf-8"))

                        toc = time.time()
                        print(f"app -2- elapsed: {(toc - tic)*1000.0} ms")

                        outputs_str = invoke_endpoint_v2(job_id, task_type, img)

                    elif task_type in ["NAMED_ENTITY_RECOGNITION"]:
                        text = body_dict["textdata"]
                        outputs_str = invoke_endpoint_v2(job_id, task_type, text)

                    else:
                        status = "Failed"
                        msg = f"unrecognized task type {task_type}"
                        outputs_str = "{}"
                        raise RuntimeError(msg)

                    # Failed to invoke endpoint
                    if outputs_str == "{}":
                        instanceId = create_auto_scaling_group(job_id, "Inference")
                        if instanceId != "":
                            taskArn = create_endpoint_v2(job_id, instanceId)

                        # Estimate ETA
                        eta = 150
                        eta_msg = ""
                        if instanceId != "":
                            eta = 90
                            ecs = boto3.client("ecs", region_name=os.environ["AWS_DEFAULT_REGION"])
                            task_id = taskTable.get_item(job_id, "InferenceTaskId")
                            if task_id and task_id != "":
                                response = ecs.describe_tasks(
                                    cluster = os.environ["CLUSTER_ARN"],
                                    tasks = [task_id]
                                )
                                tasks = response["tasks"]
                                if len(tasks) > 0:
                                    createdAt = tasks[0]["createdAt"]
                                    now = datetime.now(timezone.utc)
                                    elapsed = (now - createdAt).total_seconds()
                                    eta = max(int(eta - elapsed), 5)
                        status = "Failed"
                        msg = f"Creating a new instance for inference, {eta} seconds left."

                except Exception as e:
                    status = "Failed"
                    msg = str(e)
                    traceback.print_exc()

                toc = time.time()
                print(f"app -3- elapsed: {(toc - tic)*1000.0} ms")

                body = {"Status": status, "taskName": job_id, "Message": msg}
                print(outputs_str)
                body.update(json.loads(outputs_str))
                body_str = json.dumps(body)
                print(body_str)

                toc = time.time()
                print(f"app -4- elapsed: {(toc - tic)*1000.0} ms")

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

        elif resource == "/tasks/{task_id}/predict":

            if httpMethod == "POST":
                pathParameters = event["pathParameters"]
                job_id = unquote(pathParameters["task_id"])
                outputs_str = "{}"

                try:
                    status, msg = "Success", ""
                    body_dict = json.loads(body_str)
                    task_type = taskTable.get_item(job_id, "TaskType")

                    toc = time.time()
                    print(f"app -1- elapsed: {(toc - tic)*1000.0} ms")

                    if task_type in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION"]:
                        img_base64 = body_dict["imagedata"]
                        img_base64_split = img_base64.split(",")
                        if len(img_base64_split) != 2:
                            raise RuntimeError("invalid imagedata")
                        img_base64 = img_base64_split[1]
                        img = base64.decodebytes(img_base64.encode("utf-8"))

                        toc = time.time()
                        print(f"app -2- elapsed: {(toc - tic)*1000.0} ms")

                        outputs_str = invoke_endpoint_v2(job_id, task_type, img)

                        if outputs_str == "{}":
                            instanceId = create_auto_scaling_group(job_id, "Inference")
                            if instanceId != "":
                                taskArn = create_endpoint_v2(job_id, instanceId)
                            outputs_str = invoke_endpoint_serverless(job_id, task_type, img)

                    elif task_type in ["NAMED_ENTITY_RECOGNITION"]:
                        text = body_dict["textdata"]
                        outputs_str = invoke_endpoint_v2(job_id, task_type, text)

                        if outputs_str == "{}":
                            instanceId = create_auto_scaling_group(job_id, "Inference")
                            if instanceId != "":
                                taskArn = create_endpoint_v2(job_id, instanceId)
                            outputs_str = invoke_endpoint_serverless(job_id, task_type, text)

                    else:
                        status = "Failed"
                        msg = f"unrecognized task type {task_type}"
                        outputs_str = "{}"
                        raise RuntimeError(msg)

                except Exception as e:
                    status = "Failed"
                    msg = str(e)
                    traceback.print_exc()

                toc = time.time()
                print(f"app -3- elapsed: {(toc - tic)*1000.0} ms")

                body = {"Status": status, "taskName": job_id, "Message": msg}
                print(outputs_str)
                body.update(json.loads(outputs_str))
                body_str = json.dumps(body)
                print(body_str)

                toc = time.time()
                print(f"app -4- elapsed: {(toc - tic)*1000.0} ms")

                response = {
                    "statusCode": 200,
                    "body": body_str,
                    "headers": get_headers(),
                }

        elif resource == "/tasks/{task_id}/deploy":

            if httpMethod == "POST":

                pathParameters = event["pathParameters"]
                job_id = unquote(pathParameters["task_id"])
                endpoint_url = ""

                try:
                    status, msg = "Success", ""

                    model_uri = taskTable.get_item(job_id, "ModelUri")
                    task_type = taskTable.get_item(job_id, "TaskType")
                    task_type_s3 = task_type_map_s3[task_type]

                    if not job_id.isascii():
                        job_id_ascii = str(job_id.encode())[2:-1].replace('\\x', '')
                    else:
                        job_id_ascii = job_id

                    endpoint_config_name = f"{job_id_ascii}-{get_datetime_str()}"
                    dst_bucket = os.environ["PROD_BUCKET"]
                    dst_key = f"{task_type_s3}/{endpoint_config_name}/model.tar.gz"
                    prod_model_uri = f"s3://{dst_bucket}/{dst_key}"

                    from urllib.parse import urlparse
                    url = urlparse(model_uri, allow_fragments=False)
                    src_bucket = url.netloc
                    src_key = url.path[1:]

                    response = s3.copy_object(
                        Bucket = dst_bucket,
                        Key = dst_key,
                        CopySource = {
                            "Bucket": src_bucket,
                            "Key": src_key
                        }
                    )
                    
                    endpoint_url = create_endpoint(job_id_ascii, prod_model_uri, endpoint_config_name, task_type)

                except Exception as e:
                    status = "Failed"
                    msg = str(e)
                    traceback.print_exc()

                body = {"Status": status, "taskName": job_id, "Message": msg, "EndpointUrl": endpoint_url}
                body_str = json.dumps(body)

                statusCode = 200
                if body.get("Status", "Success") == "Failed":
                    statusCode = 500

                response = {
                    "statusCode": statusCode,
                    "body": body_str,
                    "headers": get_headers(),
                }

        elif resource == "/tasks/{task_id}/s3predict":
            if httpMethod == "POST":
                # predict with samples from a S3 directory
                pathParameters = event["pathParameters"]
                job_id = unquote(pathParameters["task_id"])
                try:
                    status, msg = "Success", ""
                except Exception as e:
                    status = "Failed"
                    msg = str(e)
                    traceback.print_exc()
                body = {"Status": status, "taskName": job_id, "Message": msg}
                body_str = json.dumps(body)

                statusCode = 200
                if body.get("Status", "Success") == "Failed":
                    statusCode = 500

                response = {
                    "statusCode": statusCode,
                    "body": body_str,
                    "headers": get_headers(),
                }

    elif resource == "aws.sagemaker":
        print("===== {}: {} =====".format(status, status_msg))

        if status == "Completed":
            artifacts = detail["ModelArtifacts"]["S3ModelArtifacts"]
            taskTable.update_item(job_id, "ModelUri", artifacts)
            # create_endpoint(job_id)

        response = {
            "statusCode": 200,
            "body": json.dumps({"TraningJobName":job_id}),
            "headers": get_headers(),
        }

    elif resource == "aws.autoscaling":
        print(f"===== AutoScaling STATUS : {status} =====")
        print(detail)
        elb = boto3.client("elbv2", region_name=os.environ["AWS_DEFAULT_REGION"])
        autoscaling = boto3.client("autoscaling", region_name=os.environ["AWS_DEFAULT_REGION"])

        if status == "Launching" and job_id != "":
            if family == "Training":
                # get bucket
                bucket = os.environ["DATA_BUCKET"]

                # get taskType
                task_type = taskTable.get_item(job_id, "TaskType")

                # get hyperparameters
                hyperparameters_str = taskTable.get_item(job_id, "HyperParameters")
                if hyperparameters_str is None:
                    hyperparameters_str = "{}"
                hyperparameters = ast.literal_eval(hyperparameters_str)

                # get instanceId
                instanceId = detail["EC2InstanceId"]

                taskId = create_training_job_v2(bucket, job_id, task_type,
                                                hyperparameters, instanceId)

            elif family == "Inference":
                # get instanceId
                instanceId = detail["EC2InstanceId"]

                TargetGroupArn = taskTable.get_item(job_id, "TargetGroupArn")
                response = elb.register_targets(
                    TargetGroupArn=TargetGroupArn,
                    Targets=[{'Id': instanceId, 'Port': 8080,},]
                )

                taskArn = create_endpoint_v2(job_id, instanceId)

        if status == "Terminating" and job_id != "":
            if family == "Training":
                response = autoscaling.delete_auto_scaling_group(
                    AutoScalingGroupName=f"MLBot-{family}-{job_id}"
                )

            elif family == "Inference":
                # get instanceId
                instanceId = detail["EC2InstanceId"]

                TargetGroupArn = taskTable.get_item(job_id, "TargetGroupArn")
                response = elb.deregister_targets(
                    TargetGroupArn=TargetGroupArn,
                    Targets=[{'Id': instanceId, 'Port': 8080,},]
                )

                # clean up autoscaling groups
                asg_name = f"MLBot-{family}-{job_id}"
                response = autoscaling.describe_auto_scaling_groups(
                    AutoScalingGroupNames = [ asg_name ],
                )
                print(response)
                AutoScalingGroups = response["AutoScalingGroups"]
                if len(AutoScalingGroups) > 0:
                    Instances = AutoScalingGroups[0]["Instances"]
                    if len(Instances) == 0:
                        # delete auto scaling group
                        response = autoscaling.delete_auto_scaling_group(
                            AutoScalingGroupName=asg_name
                        )

                        # delete load balancer
                        elb_name = f"MLBot-Inf-{job_id}"
                        response = elb.describe_load_balancers(Names = [ elb_name ])
                        LoadBalancerArn = response["LoadBalancers"][0]["LoadBalancerArn"]
                        response = elb.describe_listeners(LoadBalancerArn=LoadBalancerArn)
                        if len(response["Listeners"]) > 0:
                            ListenerArn = response["Listeners"][0]["ListenerArn"]
                            response = elb.delete_listener(ListenerArn=ListenerArn)
                        response = elb.delete_load_balancer(LoadBalancerArn = LoadBalancerArn)

                        # delete target group
                        response = elb.describe_target_groups()
                        TargetGroupDict = dict([(t["TargetGroupName"], t["TargetGroupArn"]) for t in response['TargetGroups']])
                        TargetGroupArn = TargetGroupDict[elb_name]
                        response = elb.delete_target_group(TargetGroupArn = TargetGroupArn)



        response = {
            "statusCode": 200,
            "body": json.dumps({}),
            "headers": get_headers(),
        }

    elif resource == "aws.ecs":
        print("===== ECS STATUS : {} =====".format(status))
        print(detail)
        job_id = ""

        if status == "RUNNING":
            overrides = detail["overrides"]["containerOverrides"]
            environment = overrides[0]["environment"]
            env_dict = dict([(env["name"], env["value"]) for env in environment])
            task_family = env_dict["TASK_FAMILY"]
            print("====== FAMILY: {} ======".format(task_family))

            if task_family == "Training":
                model_bucket = env_dict["MODEL_BUCKET"]
                job_id = env_dict["TASK_NAME"]
                task_id = taskArn.split("/")[-1]

                taskTable.update_item(job_id, "TrainingTaskId", task_id)
                taskTable.update_item(job_id, "TrainingDuration", "0:00:00")
                taskTable.update_item(job_id, "Status", "Training")
                taskTable.update_item(job_id, "UpdatedAt", get_utctime())

            elif task_family == "Inference":
                model_prefix = env_dict["MODEL_PREFIX"]
                job_id = env_dict["TASK_NAME"]
                task_id = taskArn.split("/")[-1]

                # Update task table and set status to Hosting
                taskTable.update_item(job_id, "InferenceTaskId", task_id)
                taskTable.update_item(job_id, "Status", "Hosting")
                taskTable.update_item(job_id, "UpdatedAt", get_utctime())

        elif status == "STOPPED":
            overrides = detail["overrides"]["containerOverrides"]
            environment = overrides[0]["environment"]
            env_dict = dict([(env["name"], env["value"]) for env in environment])
            task_family = env_dict["TASK_FAMILY"]
            print("====== FAMILY: {} ======".format(task_family))

            if task_family == "Training":
                model_bucket = env_dict["MODEL_BUCKET"]
                model_prefix = env_dict["MODEL_PREFIX"]
                model_key = "{}/model.tar.gz".format(model_prefix)
                job_id = env_dict["TASK_NAME"]
                artifact = "s3://{}/{}/model.tar.gz".format(model_bucket, model_prefix)
                taskTable.update_item(job_id, "ModelUri", artifact)
                taskTable.update_item(job_id, "ModelUrl", model_key)

                # Create new instance for inference
                instanceId = create_auto_scaling_group(job_id, "Inference")
                if instanceId != "":
                    inferenceTaskArn = create_endpoint_v2(job_id, instanceId)
                    taskTable.update_item(job_id, "InferenceTaskId", inferenceTaskArn.split("/")[-1])

                # Get task id from task table
                taskId = taskTable.get_item(job_id, "TrainingTaskId")
                elapsed = describe_task_runtime(taskId)

                # Update task table and set status to Hosting
                taskTable.update_item(job_id, "TrainingDuration", elapsed)
                taskTable.update_item(job_id, "UpdatedAt", get_utctime())

            elif task_family == "Inference":
                model_bucket = env_dict["MODEL_BUCKET"]
                job_id = env_dict["TASK_NAME"]

                # set dynamodb table record to Status="Completed"
                taskTable.update_item(job_id, "InferenceTaskId", "")
                taskTable.update_item(job_id, "Status", "Completed")
                taskTable.update_item(job_id, "UpdatedAt", get_utctime())

        response = {
            "statusCode": 200,
            "body": json.dumps({"TraningJobName": job_id}),
            "headers": get_headers(),
        }

    elif resource == "aws.apigw":
        loginDomain = os.environ["LOGIN_DOMAIN"]
        default_iss = loginDomain # "https://{}/oidc".format(loginDomain)

        print(event)

        authorizationToken = event["authorizationToken"]
        token_str = authorizationToken.split(" ")[1]
        import jwt
        token_dict = jwt.decode(token_str, options={"verify_signature": False})
        iss = token_dict["iss"]
        effect = "Deny"

        if iss == default_iss:
            effect = "Allow"

        response = {
            'principalId': 'user|a1b2c3d4',
            'policyDocument': {
                'Version': '2012-10-17',
                'Statement': [{
                    'Action': 'execute-api:Invoke',
                    'Effect': effect,
                    'Resource': ['*']
                }]
            },
            'context': {
                'key': 'value', 'number': 1, 'bool': True
            }
        }
        
    elif resource == "aws.s3control":
        job = event["job"]["id"]
        task_dict = event["tasks"][0]
        taskId = task_dict["taskId"]
        bucket_arn = task_dict["s3BucketArn"]
        key = task_dict["s3Key"]
        
        response = {
            "statusCode": 200,
            "body": json.dumps({}),
            "headers": get_headers(),
        }

    else:
        response = {
            "statusCode": 200,
            "body": json.dumps({}),
            "headers": get_headers(),
        }
    print(response)

    return response
