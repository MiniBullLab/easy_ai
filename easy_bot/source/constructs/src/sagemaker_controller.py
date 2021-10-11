import boto3
import time
import os
import ast
from iam_helper import IamHelper
from utils import get_datetime_str
from dynamodb_controller import TaskTable
from model import task_type_map, task_type_map_s3

taskTable = TaskTable()

def get_datetime_str():
    from datetime import datetime
    now = datetime.now()
    tt = now.timetuple()
    prefix = tt[0]
    name = '-'.join(['{:02}'.format(t) for t in tt[1:-3]])
    suffix = '{:03d}'.format(now.microsecond)[:3]
    job_name_suffix = "{}-{}-{}".format(prefix, name, suffix)
    return job_name_suffix

def create_training_job(bucket, jobname, task, hyperparameters):
    s3_path = "s3://{}/{}/".format(bucket, jobname)
    helper = IamHelper
    client = boto3.client("sagemaker")
    job_name = "ml-bot-{}-{}".format(task.replace("_", "-"), get_datetime_str())
    account_id = IamHelper.get_account_id()
    region = IamHelper.get_region()
    partition = IamHelper.get_partition()
    role_name = "AmazonSageMaker-ExecutionRole-20200512T121482"
    role_arn = "arn:{}:iam::{}:role/service-role/{}".format(partition, account_id, role_name)
    s3_output_path = 's3://sagemaker-{}-{}/'.format(region, account_id)
    response = client.create_training_job(
        TrainingJobName = job_name,
        HyperParameters = {},
        AlgorithmSpecification={
            'TrainingImage': f'{account_id}.dkr.ecr.{region}.amazonaws.com.cn/ml-bot-training',
            'TrainingInputMode': 'File',
        },
        RoleArn = role_arn,
        InputDataConfig=[
            {
                'ChannelName': "training", # environment variable SM_CHANNEL_TRAINING and /opt/ml/input/data/training
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': s3_path,
                    },
                },
            },
        ],
        OutputDataConfig={
            'S3OutputPath': s3_output_path
        },
        ResourceConfig = {
            'InstanceType': 'ml.g4dn.xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 64,
        },
        StoppingCondition = {
            'MaxRuntimeInSeconds': 600,
        },
    )
    return response

def create_endpoint(job_name, model_uri, endpoint_config_name, task_type):
    ecs = boto3.client("ecs")

    print("Creating SageMaker Endpoint {} from {}".format(job_name, model_uri))
    
    client = boto3.client("sagemaker")
    region = IamHelper.get_region()
    partition = IamHelper.get_partition()
    account_id = IamHelper.get_account_id()
    role_arn = os.environ["SAGEMAKER_EXEC_ROLE_ARN"]

    task_type_s3 = task_type_map_s3[task_type]

    # describe task definition
    response = ecs.describe_task_definition(
        taskDefinition = os.environ["INFERENCE_TASK_ARN"],
    )
    taskDefinition = response["taskDefinition"]
    containerDefinitions = taskDefinition["containerDefinitions"]
    ecr_image_uri = containerDefinitions[0]["image"]

    # get hyperparameters
    hyperparameters_str = taskTable.get_item(job_name, "HyperParameters")
    if hyperparameters_str is None:
        hyperparameters_str = "{}"
    hyperparameters = ast.literal_eval(hyperparameters_str)
    
    # create model
    response = client.create_model(
        ModelName = endpoint_config_name,
        PrimaryContainer = {
            "Image": ecr_image_uri,
            "ModelDataUrl": model_uri,
            'Environment': {
                'MODEL_BUCKET': os.environ["MODEL_BUCKET"],
                'MODEL_PREFIX': "{}/{}".format(task_type_s3, job_name),
                'TASK_TYPE': task_type,
                'AWS_DEFAULT_REGION': region,
                'HOSTING_MODE': "SAGEMAKER",
                'MODEL_SERVER_WORKERS': '2',
                'HYPERPARAMETERS': str(hyperparameters),
            },
        },
        ExecutionRoleArn = role_arn
    )

    # create endpoint config
    response = client.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': endpoint_config_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.g4dn.xlarge',
            },
        ]
    )

    # create endpoint
    response = client.list_endpoints(NameContains = job_name)
    endpoints = [m["EndpointName"] for m in response.get("Endpoints", [])]
    if job_name in endpoints:
        response = client.describe_endpoint(EndpointName = job_name)
        status = response["EndpointStatus"]
        if status in ["Failed"]:
            response = client.delete_endpoint(EndpointName = job_name)
            time.sleep(10)
            response = client.create_endpoint(
                EndpointName = job_name,
                EndpointConfigName = endpoint_config_name
            )
        else:            
            response = client.update_endpoint(
                EndpointName = job_name,
                EndpointConfigName = endpoint_config_name
            )
    else:
        response = client.create_endpoint(
            EndpointName = job_name,
            EndpointConfigName = endpoint_config_name
        )
    if region in ["cn-north-1", "cn-northwest-1"]:
        endpoint_url = f"https://console.amazonaws.cn/sagemaker/home?region={region}#/endpoints/{job_name}"
    else:
        endpoint_url = f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{job_name}"
    return endpoint_url

def describe_training_job(job):
    helper = IamHelper
    client = boto3.client("sagemaker")

    training_jobs = client.list_training_jobs(MaxResults=100).get("TrainingJobSummaries", [])
    names = [j['TrainingJobName'] for j in training_jobs]
    stats = []
    # print("names={}".format(str(names)))
    
    if job in names:
        response = client.describe_training_job(TrainingJobName=job)
        transitions = response["SecondaryStatusTransitions"]
        # status = transitions[-1].get("Status", "Unknown")
        stats = [t.get("Status", "") for t in transitions]

    artifacts = ""
    if "Completed" in stats:
        artifacts = describe_training_job_artifact(job_id)

    return stats, artifacts

def describe_training_job_artifact(job):
    helper = IamHelper
    client = boto3.client("sagemaker")

    training_jobs = client.list_training_jobs(MaxResults=100).get("TrainingJobSummaries", [])
    names = [j['TrainingJobName'] for j in training_jobs]
    artifacts = ""
    # print("names={}".format(str(names)))
    
    if job in names:
        response = client.describe_training_job(TrainingJobName=job)
        artifacts = response["ModelArtifacts"]["S3ModelArtifacts"]

    return artifacts

def describe_endpoint(job):
    client = boto3.client("sagemaker")
    endpoints = client.list_endpoints()
    names = [endpoint["EndpointName"] for endpoint in endpoints["Endpoints"]]
    stats = [endpoint["EndpointStatus"] for endpoint in endpoints["Endpoints"]]
    status = "NotStarted"
    if job in names:
        for idx, name in enumerate(names):
            if job == name:
                status = stats[idx]
    return status

def list_endpoints(job):
    client = boto3.client("sagemaker")
    endpoints = client.list_endpoints()
    names = [endpoint["EndpointName"] for endpoint in endpoints["Endpoints"]]
    stats = [endpoint["EndpointStatus"] for endpoint in endpoints["Endpoints"]]
    return [{"Name": n, "Status": s} for n, s in zip(names, stats)]

def invoke_endpoint(body_str):
    region = IamHelper.get_region()

    session = Session(region_name = f"{region}")
    runtime = session.client("runtime.sagemaker")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="image/jpeg",
        Body=img,  # json.dumps(data),
    )
    print(response)
    outputs = response["Body"].read()
    outputs_str = outputs.decode("utf-8")
    outputs_dict = json.loads(outputs_str)
    outputs_dict.update(Status="Success")
    outputs_str = json.dumps(outputs_dict)

    print(outputs_str)
    
