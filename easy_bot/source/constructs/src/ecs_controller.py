import boto3
import os
import sys
from iam_helper import IamHelper
import json
from datetime import timedelta
from dynamodb_controller import TaskTable
from utils import get_utctime
from model import task_type_map, task_type_map_s3, task_type_lut
import time
from s3_controller import list_objects, get_object_url
from elb_controller import create_load_balancer
import asyncio
import functools
import ast
from urllib.parse import unquote
import traceback

thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(thisdir, "lambda_dependencies"))
import requests

taskTable = TaskTable()
autoscaling = boto3.client("autoscaling", region_name=os.environ["AWS_DEFAULT_REGION"])
ecs = boto3.client("ecs", region_name=os.environ["AWS_DEFAULT_REGION"])
cloudwatch = boto3.client("cloudwatch", region_name=os.environ["AWS_DEFAULT_REGION"])
elb = boto3.client("elbv2", region_name=os.environ["AWS_DEFAULT_REGION"])

class Cluster():
    def __init__():
        self.cluster = os.environ["CLUSTER_ARN"]
        self.dataBucket = os.environ["DATA_BUCKET"]
        self.modelBucket = os.environ["MODEL_BUCKET"]
        self.trainingTaskArn = os.environ["TRAINING_TASK_ARN"]


def get_last_status(task_id):
    response = ecs.describe_tasks(
        cluster = os.environ["CLUSTER_ARN"],
        tasks = [task_id,]
    )
    tasks = response["tasks"]
    if len(tasks) == 0:
        return None
    lastStatus = tasks[0]["lastStatus"]
    return lastStatus

def get_cloudwatch_logs(task_id):
    response = ecs.describe_tasks(
        cluster = os.environ["CLUSTER_ARN"],
        tasks = [task_id,]
    )
    tasks = response["tasks"]
    if len(tasks) == 0:
        return None
    taskDefinitionArn = tasks[0]["taskDefinitionArn"]

    cloudwatch_logs_url = ""
    response = ecs.describe_task_definition(
        taskDefinition = taskDefinitionArn
    )
    taskDefinition = response["taskDefinition"]
    containerDefinitions = taskDefinition["containerDefinitions"]
    container = containerDefinitions[0]
    containerName = container["name"]
    logConfiguration = container["logConfiguration"]
    options = logConfiguration["options"]
    awslogs_group = options["awslogs-group"]
    awslogs_region = options["awslogs-region"]
    awslogs_stream_prefix = options["awslogs-stream-prefix"]
    if awslogs_region.startswith("cn"):
        cloudwatch_logs_url = f"https://{awslogs_region}.console.amazonaws.cn/cloudwatch/home?region={awslogs_region}#logEventViewer:group={awslogs_group};stream={awslogs_stream_prefix}/{containerName}/{task_id}"
    else:
        cloudwatch_logs_url = f"https://{awslogs_region}.console.aws.amazon.com/cloudwatch/home?region={awslogs_region}#logEventViewer:group={awslogs_group};stream={awslogs_stream_prefix}/{containerName}/{task_id}"

    return cloudwatch_logs_url

def create_auto_scaling_group(job_id, family):
    """
    family: either "Training" or "Inference"
    """
    ref_asg_name = os.environ[f"{family.upper()}_ASG"]
    response = autoscaling.describe_auto_scaling_groups(
        AutoScalingGroupNames = [ref_asg_name,]
    )
    if len(response["AutoScalingGroups"]) == 0:
        raise RuntimeError(f"Failed to locate {ref_asg_name} AutoScaling Group.")
    LaunchConfigurationName = response["AutoScalingGroups"][0]["LaunchConfigurationName"]

    asg_name = f"MLBot-{family}-{job_id}"

    # create elb
    if family in ["Inference"]:
        elb_name = f"MLBot-Inf-{job_id}"
        create_load_balancer(job_id, elb_name)

    # create autoscaling group
    response = autoscaling.describe_auto_scaling_groups(
        AutoScalingGroupNames=[asg_name,]
    )
    print(response)
    asg_kwargs = {
        "AutoScalingGroupName": asg_name,
        "LaunchConfigurationName": LaunchConfigurationName,
        "MinSize": 0, "MaxSize": 1, "DesiredCapacity": 1,
    }
    # if family in ["Inference"]:
    #     asg_kwargs.update(LoadBalancerNames = [asg_name])
    if len(response["AutoScalingGroups"]) > 0:
        response = autoscaling.update_auto_scaling_group(**asg_kwargs,
            VPCZoneIdentifier = os.environ["PRIVATE_SUBNETS"],
        )
    else:
        response = autoscaling.create_auto_scaling_group(**asg_kwargs,
            VPCZoneIdentifier = os.environ["PRIVATE_SUBNETS"],
        )

    response = autoscaling.put_scaling_policy(
        AutoScalingGroupName = asg_name,
        PolicyName='DownScalingPolicy',
        PolicyType='SimpleScaling',
        AdjustmentType='ExactCapacity',
        ScalingAdjustment=0,
    )
    PolicyARN = response['PolicyARN']

    EvaluationPeriods = 1
    if family == "Inference":
        EvaluationPeriods = 12

    response = cloudwatch.put_metric_alarm(
        AlarmName = asg_name,
        AlarmDescription = f"Metric alarm for scaling down on {asg_name} AutoScaling group.",
        AlarmActions = [
            PolicyARN,
        ],
        MetricName = "CPUUtilization",
        Namespace = "AWS/EC2",
        Statistic = "Average",
        Dimensions = [{
            'Name': 'AutoScalingGroupName',
            'Value': asg_name
        },],
        ComparisonOperator = "LessThanThreshold",
        Threshold = 1.0,
        EvaluationPeriods = EvaluationPeriods,
        Period = 300,
    )

    instanceId = ""
    for it in range(2):
        response = autoscaling.describe_auto_scaling_groups(
            AutoScalingGroupNames=[asg_name,]
        )
        if len(response["AutoScalingGroups"]) > 0:
            asg_status = response["AutoScalingGroups"][0]
            instances = asg_status["Instances"]
            if len(instances) > 0:
                for instance in instances:
                    LifecycleState = instance["LifecycleState"]
                    if LifecycleState == "InService" and instance["InstanceId"] != "":
                        instanceId = instance["InstanceId"]
                        break
            if instanceId == "":
                print(f"No instance found in {asg_name}, waiting 1 second, waited {it} times.")
                time.sleep(1.0)
            else:
                break

    return instanceId

def create_training_job_v2(bucket, job_id, task_type, hyperparameters, instanceId):
    print("Creating training job {} with {} {}".format(job_id, task_type, hyperparameters))
    task_type_s3 = task_type_map_s3[task_type]
    helper = IamHelper
    client = boto3.client("ecs")
    data_prefix = "{}/{}".format(task_type_s3, job_id)

    if task_type in ["IMAGE_CLASSIFICATION"]:
        classes = taskTable.get_item(job_id, "Classes")
        print(f"CLASSES={classes}")
        classes = ast.literal_eval(classes) # convert str to list
        print(f"CLASSES={classes}")
        if classes:
            hyperparameters.update(CLASSES=classes)
    elif task_type in ["OBJECT_DETECTION"]:
        data_prefix = taskTable.get_item(job_id, "DataPrefix")
        if data_prefix == "":
            data_prefix = "{}/{}".format(task_type_s3, job_id)

    response = client.run_task(
        launchType = 'EC2',
        taskDefinition = os.environ["TRAINING_TASK_ARN"],
        cluster = os.environ["CLUSTER_ARN"],
        overrides={ 'containerOverrides': [{
            'name': 'trainingContainer',
            'environment': [
                { 'name': 'DATA_BUCKET', 'value': os.environ["DATA_BUCKET"] },
                { 'name': 'DATA_PREFIX', 'value': data_prefix },
                { 'name': 'MODEL_BUCKET', 'value': os.environ["MODEL_BUCKET"] },
                { 'name': 'MODEL_PREFIX', 'value': "{}/{}".format(task_type_s3, job_id) },
                { 'name': 'TASK_FAMILY', 'value': "Training" },
                { 'name': 'TASK_TYPE', 'value': task_type },
                { 'name': 'TASK_NAME', 'value': job_id },
                { 'name': 'HYPERPARAMETERS', 'value': str(hyperparameters) },
            ],
        }], },
        placementConstraints=[
            { 'type': 'memberOf', 'expression': f"ec2InstanceId == {instanceId}" },
        ],
    )
    print(response)
    tasks = response["tasks"]

    taskId = ""
    if len(tasks) > 0:
        task0 = tasks[0]
        container0 = task0["containers"][0]
        taskArn = container0["taskArn"]
        taskId = taskArn.split("/")[-1]
    else:
        raise RuntimeError("Failed to start the training task due to limited container instances.")
    return taskId


def create_training_job(bucket, job_id, task_type, hyperparameters):
    print("Creating training job {} with {} {}".format(job_id, task_type, hyperparameters))
    task_type_s3 = task_type_map_s3[task_type]
    helper = IamHelper
    client = boto3.client("ecs")
    data_prefix = "{}/{}".format(task_type_s3, job_id)

    if task_type in ["IMAGE_CLASSIFICATION"]:
        classes = taskTable.get_item(job_id, "Classes")
        print(f"CLASSES={classes}")
        classes = ast.literal_eval(classes) # convert str to list
        print(f"CLASSES={classes}")
        if classes:
            hyperparameters.update(CLASSES=classes)
    elif task_type in ["OBJECT_DETECTION"]:
        data_prefix = taskTable.get_item(job_id, "DataPrefix")
        if data_prefix == "":
            data_prefix = "{}/{}".format(task_type_s3, job_id)

    response = client.run_task(
        launchType = 'EC2',
        taskDefinition = os.environ["TRAINING_TASK_ARN"],
        cluster = os.environ["CLUSTER_ARN"],
        overrides={ 'containerOverrides': [{
            'name': 'trainingContainer',
            'environment': [
                { 'name': 'DATA_BUCKET', 'value': os.environ["DATA_BUCKET"] },
                { 'name': 'DATA_PREFIX', 'value': data_prefix },
                { 'name': 'MODEL_BUCKET', 'value': os.environ["MODEL_BUCKET"] },
                { 'name': 'MODEL_PREFIX', 'value': "{}/{}".format(task_type_s3, job_id) },
                { 'name': 'TASK_FAMILY', 'value': "Training" },
                { 'name': 'TASK_TYPE', 'value': task_type },
                { 'name': 'TASK_NAME', 'value': job_id },
                { 'name': 'HYPERPARAMETERS', 'value': str(hyperparameters) },
            ],
        }], },
        placementConstraints=[
            { 'type': 'memberOf', 'expression': 'attribute:ecs.instance-type =~ g4dn.2xlarge' },
        ],
    )
    print(response)
    tasks = response["tasks"]

    taskId = ""
    if len(tasks) > 0:
        task0 = tasks[0]
        container0 = task0["containers"][0]
        taskArn = container0["taskArn"]
        taskId = taskArn.split("/")[-1]
    else:
        raise RuntimeError("Failed to start the training task due to limited container instances.")
    return taskId


def describe_training_job(job_id):
    ecs = boto3.client("ecs")

    # Get training task id
    task_id = taskTable.get_item(job_id, "TrainingTaskId")
    print("JobId={}, TrainingTaskId={}".format(job_id, task_id))
    if task_id is None or task_id is "":
        return [], "", "" # training job not started yet

    # Get status of the training task
    lastStatus = get_last_status(task_id)
    artifacts = []
    cloudwatch_logs_url = get_cloudwatch_logs(task_id)

    if lastStatus is None:
        taskTable.update_item(job_id, "Status", "Completed")
        lastStatus = "STOPPED"

    if lastStatus in ["PROVISIONING", "PENDING"]:
        return ["Starting"], artifacts, cloudwatch_logs_url
    elif lastStatus in ["RUNNING", "STOPPING"]:
        return ["Starting", "Downloading"], artifacts, cloudwatch_logs_url
    elif lastStatus in ["STOPPED"]:
        model_bucket = os.environ["MODEL_BUCKET"]
        model_key = taskTable.get_item(job_id, "ModelUrl")
        if model_key and model_key != "":
            model_url = get_object_url(model_bucket, model_key)
            if model_url is not None:
                artifacts.append(model_url)
        return ["Starting", "Downloading", "Training", "Uploading", "Completed"], artifacts, cloudwatch_logs_url
    else:
        return [], artifacts, cloudwatch_logs_url


def run_task(job_id):
    print("Creating inference endpoint {}".format(job_id))
    helper = IamHelper
    ecs = boto3.client("ecs")

    # stop inference endpoints
    response = ecs.list_tasks(
        cluster = os.environ["CLUSTER_ARN"],
        family = os.environ["INFERENCE_FAMILY"]
    )
    taskArns = response["taskArns"]
    ###### WARNING: stopping all inference endpoints ######
    for task_id in taskArns:
        ecs.stop_task(
            cluster = os.environ["CLUSTER_ARN"],
            task = task_id,
            reason = "clean up to host new inference endpoints"
        )
    ###### WARNING: stopping all inference endpoints ######

    # get task type from task table
    task_type = taskTable.get_item(job_id, "TaskType")
    task_type_s3 = task_type_map_s3[task_type]

    # get task type from task table
    model_uri = taskTable.get_item(job_id, "ModelUri")

    # create new inference endpoint
    response = ecs.run_task(
        launchType = 'EC2',
        taskDefinition = os.environ["INFERENCE_TASK_ARN"],
        cluster = os.environ["CLUSTER_ARN"],
        overrides={ 'containerOverrides': [{
            'name': 'inferenceContainer',
            'environment': [
                { 'name': 'DATA_BUCKET', 'value': os.environ["DATA_BUCKET"] },
                { 'name': 'DATA_PREFIX', 'value': "{}/{}".format(task_type_s3, job_id) },
                { 'name': 'MODEL_BUCKET', 'value': os.environ["MODEL_BUCKET"] },
                { 'name': 'MODEL_PREFIX', 'value': "{}/{}".format(task_type_s3, job_id) },
                { 'name': 'MODEL_URI', 'value': model_uri },
                { 'name': 'TASK_FAMILY', 'value': "Inference" },
                { 'name': 'TASK_TYPE', 'value': task_type },
                { 'name': 'TASK_NAME', 'value': job_id },
                { 'name': 'MODEL_SERVER_WORKERS', 'value': '2' },
            ],
        }], },
        placementConstraints=[
            { 'type': 'memberOf', 'expression': 'attribute:ecs.instance-type =~ g4dn.xlarge' },
        ],
    )
    print(response)

    tasks = response["tasks"]
    taskArn = ""
    if len(tasks) > 0:
        task0 = tasks[0]
        containers = task0["containers"]
        container0 = containers[0]
        taskArn = container0["taskArn"]
    return taskArn


def describe_endpoint(job_id):
    ecs = boto3.client("ecs")

    task_id = taskTable.get_item(job_id, "InferenceTaskId")
    if task_id is None or task_id == "":
        status = "Creating"
        task_id = ""
    else:
        lastStatus = get_last_status(task_id)
        if lastStatus == "RUNNING":
            status = "InService"
        else:
            status = "NotStarted"
    return status, task_id


def create_endpoint_v2(job_id, instanceId):
    print("Creating inference endpoint {}".format(job_id))

    # get task type from task table
    task_type = taskTable.get_item(job_id, "TaskType")
    task_type_s3 = task_type_map_s3[task_type]

    # get task type from task table
    model_uri = taskTable.get_item(job_id, "ModelUri")

    # get hyperparameters from task table
    hyperparameters_str = taskTable.get_item(job_id, "HyperParameters")
    if hyperparameters_str is None:
        hyperparameters_str = "{}"
    hyperparameters = ast.literal_eval(hyperparameters_str)

    # create new inference endpoint
    response = ecs.run_task(
        launchType = 'EC2',
        taskDefinition = os.environ["INFERENCE_TASK_ARN"],
        cluster = os.environ["CLUSTER_ARN"],
        overrides={ 'containerOverrides': [{
            'name': 'inferenceContainer',
            'environment': [
                { 'name': 'DATA_BUCKET', 'value': os.environ["DATA_BUCKET"] },
                { 'name': 'DATA_PREFIX', 'value': "{}/{}".format(task_type_s3, job_id) },
                { 'name': 'MODEL_BUCKET', 'value': os.environ["MODEL_BUCKET"] },
                { 'name': 'MODEL_PREFIX', 'value': "{}/{}".format(task_type_s3, job_id) },
                { 'name': 'MODEL_URI', 'value': model_uri },
                { 'name': 'TASK_FAMILY', 'value': "Inference" },
                { 'name': 'TASK_TYPE', 'value': task_type },
                { 'name': 'TASK_NAME', 'value': job_id },
                { 'name': 'MODEL_SERVER_WORKERS', 'value': '2' },
                { 'name': 'HYPERPARAMETERS', 'value': str(hyperparameters) },
            ],
        }], },
        placementConstraints=[
            { 'type': 'memberOf', 'expression': f"ec2InstanceId == {instanceId}" },
        ],
    )
    print(response)

    tasks = response["tasks"]
    taskArn = ""
    if len(tasks) > 0:
        task0 = tasks[0]
        containers = task0["containers"]
        container0 = containers[0]
        taskArn = container0["taskArn"]

    ##### get private ip address of the instance
    # ec2 = boto3.client("ec2")
    # response = ec2.describe_instances(InstanceIds=[instanceId])
    # Reservation0 = response["Reservations"][0]
    # Instance0 = Reservation0["Instances"][0]
    # NetworkInterface0 = Instance0["NetworkInterfaces"][0]
    # PrivateIpAddress = NetworkInterface0["PrivateIpAddress"]
    # url = "http://{}:8080/invocations".format(PrivateIpAddress)
    # taskTable.update_item(job_id, "InferenceEndpoint", url)

    return taskArn


def invoke_endpoint_v2(job_id, task_type, data_json):
    ecs = boto3.client("ecs")

    tic = time.time()

    url = taskTable.get_item(job_id, "InferenceEndpoint")
    outputs_str = "{}"

    if url and url != "":
        if task_type in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION"]:
            data_bytes = data_json
            headers = {'Content-type': 'image/jpeg'}
        elif task_type in ["NAMED_ENTITY_RECOGNITION"]:
            data_json = json.dumps({"data": data_json})
            headers = {'Content-type': 'application/json'}
            data_bytes = data_json
        else:
            headers = {'Content-type': 'application/json'}

        toc = time.time()
        print(f"invoke_endpoint -2- elapsed: {(toc - tic)*1000.0} ms")

        wait_count = 0
        outputs_dict = {}

        print(f"INVOCATIONS: {url}")
        for it in range(1):
            try:
                response = requests.post(url, data=data_bytes, headers=headers, timeout=2.0)

                if hasattr(response, 'text'):
                    outputs = response.text
                    outputs_str = outputs

                    if outputs_str.startswith("{"):
                        outputs_dict = json.loads(outputs_str)

                        toc = time.time()
                        print(f"invoke_endpoint -3- elapsed: {(toc - tic)*1000.0} ms")

                        outputs_dict.update(Status="Success")
                        outputs_str = json.dumps(outputs_dict)
                        break
                    else:
                        print(f"got {outputs_str} as outputs_str, performing another attempt.")
                        outputs_str = "{}"
                        time.sleep(0.1)
                else:
                    print("POST reponse got no text attribute, performin another attempt.")
                    time.sleep(0.1)

            except Exception as e:
                print(f"failed to send POST event to {url}, performing another attempt")
                time.sleep(0.1)

    return outputs_str


def invoke_endpoint(job_id, task_type, data_json):
    ecs = boto3.client("ecs")

    tic = time.time()

    if False:
        url = taskTable.get_item(job_id, "InferenceEndpoint")
        if url and url != "":
            if task_type in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION"]:
                data_bytes = data_json
                headers = {'Content-type': 'image/jpeg'}
            elif task_type in ["NAMED_ENTITY_RECOGNITION"]:
                data_json = json.dumps({"data": data_json})
                headers = {'Content-type': 'application/json'}
                data_bytes = data_json
            else:
                headers = {'Content-type': 'application/json'}

            toc = time.time()
            print(f"invoke_endpoint -2- elapsed: {(toc - tic)*1000.0} ms")

            wait_count = 0
            outputs_dict = {}

            print(f"INVOCATIONS: {url}")
            try:
                response = requests.post(url, data=data_bytes, headers=headers, timeout=2.0)

                outputs_str = "{}"
                if hasattr(response, 'text'):
                    outputs = response.text
                    outputs_str = outputs

                if outputs_str.startswith("{"):
                    outputs_dict = json.loads(outputs_str)

                    toc = time.time()
                    print(f"invoke_endpoint -3- elapsed: {(toc - tic)*1000.0} ms")

                    outputs_dict.update(Status="Success")
                    outputs_str = json.dumps(outputs_dict)
                    return outputs_str
            except:
                print(f"failed to send POST event to {url}")
                pass

    task_id = taskTable.get_item(job_id, "InferenceTaskId")

    # get instance id of the inference endpoint
    response = ecs.describe_tasks(
        cluster = os.environ["CLUSTER_ARN"],
        tasks = [task_id,]
    )
    print(response)
    tasks = response["tasks"]
    if len(tasks) == 0:
        # Cann't find the task_id in the ECS task list,
        # this might due to the task is STOPPED and is timeout for living the in ECS task list.
        # The following behavior would launch a new inference endpoint
        task_id = run_task(job_id)
        taskTable.update_item(job_id, "InferenceTaskId", task_id)
        response = ecs.describe_tasks(
            cluster = os.environ["CLUSTER_ARN"],
            tasks = [task_id,]
        )
        tasks = response["tasks"]

    task0 = tasks[0]

    toc = time.time()
    print(f"invoke_endpoint -0- elapsed: {(toc - tic)*1000.0} ms")

    def _wait_for_task_running(_task_id):
        lastStatus = get_last_status(_task_id)
        while lastStatus != "RUNNING":
            time.sleep(1)
            lastStatus = get_last_status(_task_id)

    # check whether the inference endpoint has been stopped
    containers = task0["containers"]
    container0 = containers[0]
    lastStatus = container0["lastStatus"]
    if lastStatus == "STOPPED":
        # The task has been STOPPED
        task_id = run_task(job_id)
        taskTable.update_item(job_id, "InferenceTaskId", task_id)
        response = ecs.describe_tasks(
            cluster = os.environ["CLUSTER_ARN"],
            tasks = [task_id,]
        )
        tasks = response["tasks"]
        task0 = tasks[0]
        _wait_for_task_running(task_id)

    toc = time.time()
    print(f"invoke_endpoint -1- elapsed: {(toc - tic)*1000.0} ms")

    # obtain instance id from the response
    containerInstanceArn = task0["containerInstanceArn"]
    response = ecs.describe_container_instances(
        cluster = os.environ["CLUSTER_ARN"],
        containerInstances = [containerInstanceArn,]
    )
    containerInstances = response["containerInstances"]
    containerInstance0 = containerInstances[0]
    ec2InstanceId = containerInstance0["ec2InstanceId"]

    # get private ip address of the instance
    ec2 = boto3.client("ec2")
    response = ec2.describe_instances(InstanceIds=[ec2InstanceId])
    Reservation0 = response["Reservations"][0]
    Instance0 = Reservation0["Instances"][0]
    NetworkInterface0 = Instance0["NetworkInterfaces"][0]
    PrivateIpAddress = NetworkInterface0["PrivateIpAddress"]

    url = "http://{}:8080/invocations".format(PrivateIpAddress)
    print(url)

    task_type = taskTable.get_item(job_id, "TaskType")
    if task_type in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION"]:
        data_bytes = data_json
        headers = {'Content-type': 'image/jpeg'}
    elif task_type in ["NAMED_ENTITY_RECOGNITION"]:
        data_json = json.dumps({"data": data_json})
        headers = {'Content-type': 'application/json'}
        data_bytes = data_json
    else:
        headers = {'Content-type': 'application/json'}

    toc = time.time()
    print(f"invoke_endpoint -2- elapsed: {(toc - tic)*1000.0} ms")

    wait_count = 0
    outputs_dict = {}
    while True:
        print(f"INVOCATIONS: {url}")
        response = requests.post(url, data=data_bytes, headers=headers)

        outputs_str = "{}"
        if hasattr(response, 'text'):
            outputs = response.text
            outputs_str = outputs

        if outputs_str.startswith("{"):
            outputs_dict = json.loads(outputs_str)
            taskTable.update_item(job_id, "InferenceEndpoint", url)
            break
        elif outputs_str.startswith("<"):
            print(f"Wait 1 second, waited {wait_count} times.")
            if wait_count > 8:
                raise RuntimeError("Timeout to download inference docker container, please wait a minute and have another attempt.")
            time.sleep(1.0)
            wait_count += 1
            continue

    toc = time.time()
    print(f"invoke_endpoint -3- elapsed: {(toc - tic)*1000.0} ms")

    outputs_dict.update(Status="Success")
    outputs_str = json.dumps(outputs_dict)
    return outputs_str


def describe_task_runtime(task_arn):
    print(f"describe_tasks: task_arn={task_arn}")
    ecs = boto3.client("ecs")
    response = ecs.describe_tasks(
        cluster = os.environ["CLUSTER_ARN"],
        tasks = [task_arn,]
    )
    tasks = response["tasks"]
    if len(tasks)==0:
        return str(timedelta(seconds=0))
    task0 = tasks[0]
    startedAt = task0["startedAt"]
    stoppedAt = task0["stoppedAt"]
    elapsed = stoppedAt - startedAt
    print("startedAt: {}, stoppedAt: {}, elapsed: {}".format(startedAt, stoppedAt, elapsed))
    return str(timedelta(seconds=elapsed.seconds))


def list_tasks():
    print("===== LIST_TASKS =====")

    # scan task table for items
    items = taskTable.scan()
    taskNames = [item["JobId"]["S"] for item in items]
    taskTypes = [item.get("TaskType", {"S":""})["S"] for item in items]
    statusNames = [item.get("Status", {"S": ""})["S"] for item in items]
    taskRuntimes = [item.get("TrainingDuration", {"S": "0:00:00"})["S"] for item in items]
    createTimes = [item.get("CreatedAt", {"S":""})["S"] for item in items]
    updateTimes = [item.get("UpdatedAt", {"S":""})["S"] for item in items]

    return taskNames, taskTypes, taskRuntimes, statusNames, createTimes, updateTimes, None


def stop_task(job_id):
    print("===== STOP_TASK: {} =====".format(job_id))

    # get inference task id from task table
    task_id = taskTable.get_item(job_id, "InferenceTaskId")

    # stop inference endpoint
    ecs = boto3.client("ecs")
    ecs.stop_task(
        cluster = os.environ["CLUSTER_ARN"],
        task = task_id,
        reason = "request to stop inference endpoint from user interface"
    )

    # update time stamp in task table
    taskTable.update_item(job_id, "UpdatedAt", get_utctime())

    return "Success", ""


def delete_task(job_id):
    print("===== DELETE_TASK: {} =====".format(job_id))
    s3 = boto3.client("s3")
    msg = ""

    task_type = taskTable.get_item(job_id, "TaskType")

    if task_type:
        task_type_s3 = task_type_map_s3[task_type]

        # delete objects
        dataBucket = os.environ["DATA_BUCKET"]
        prefix = f"{task_type_s3}/{job_id}/"
        objs = list_objects(dataBucket, prefix)
        objs = [{'Key': obj} for obj in objs]
        f = lambda A, n=200: [A[i:i+n] for i in range(0, len(A), n)]
        subsets = f(objs)

        # parallel logic for running delete_objects
        async def main():
            loop = asyncio.get_running_loop()
            # src_bucket = os.environ["SRC_BUCKET"]
            objects = await asyncio.gather(
                *[
                    loop.run_in_executor(None, functools.partial(s3.delete_objects, Bucket=dataBucket, Delete={"Objects": subset, "Quiet": True}))
                    for subset in subsets
                ]
            )
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())

        msg = f"removed {len(objs)} keys in data bucket with prefix {prefix}."

    # delete the record in task table
    taskTable.delete_item(job_id)

    return "Success", msg

