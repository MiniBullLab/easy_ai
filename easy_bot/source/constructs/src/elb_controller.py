import boto3
import os
import time
from dynamodb_controller import TaskTable

taskTable = TaskTable()

elb = boto3.client("elbv2", region_name=os.environ["AWS_DEFAULT_REGION"])

def create_load_balancer(job_id, elb_name):
    # create target group
    response = elb.describe_target_groups()
    TargetGroupNames = [t["TargetGroupName"] for t in response["TargetGroups"]]
    if elb_name not in TargetGroupNames:
        response = elb.create_target_group(
            Name=elb_name,
            Protocol='HTTP',
            ProtocolVersion='HTTP1',
            Port=8080,
            VpcId=os.environ["VPC_ID"],
            HealthCheckProtocol='HTTP',
            HealthCheckPort='8080',
            HealthCheckEnabled=True,
            HealthCheckPath='/ping',
            HealthCheckIntervalSeconds=5,
            HealthCheckTimeoutSeconds=2,
            HealthyThresholdCount=2,
            UnhealthyThresholdCount=2,
            TargetType='instance',
        )

    # create load balancer
    response = elb.describe_load_balancers()
    elb_kwargs = {
        "Name": elb_name,
        "Subnets": os.environ["PRIVATE_SUBNETS"].split(","),
        "SecurityGroups": [ os.environ["INFERENCE_SG"] ],
        "Scheme": "internal",
        "Type": 'application',
    }
    LoadBalancerNames = [d["LoadBalancerName"] for d in response["LoadBalancers"]]
    if elb_name not in LoadBalancerNames:
        response = elb.create_load_balancer(**elb_kwargs)

    # create listener
    response = elb.describe_load_balancers(Names = [elb_name,])
    LoadBalancerArn = response["LoadBalancers"][0]["LoadBalancerArn"]
    DNSName = response["LoadBalancers"][0]["DNSName"]
    response = elb.describe_target_groups()
    TargetGroupDict = dict([(t["TargetGroupName"], t["TargetGroupArn"]) for t in response['TargetGroups']])
    TargetGroupArn = TargetGroupDict[elb_name]
    taskTable.update_item(job_id, "TargetGroupArn", TargetGroupArn)
    response = elb.describe_listeners(LoadBalancerArn=LoadBalancerArn)
    listener_kwargs = {
        "LoadBalancerArn": LoadBalancerArn, "Protocol": 'HTTP', "Port": 8080,
        "DefaultActions": [{ 'Type': 'forward', 'TargetGroupArn': TargetGroupArn }]
    }
    if len(response["Listeners"]) == 0:
        print(response)
        response = elb.create_listener(**listener_kwargs)

    url = "http://{}:8080/invocations".format(DNSName)
    taskTable.update_item(job_id, "InferenceEndpoint", url)
