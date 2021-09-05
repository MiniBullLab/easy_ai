import boto3
import os
import sys
from iam_helper import IamHelper
import json

dynamodb = boto3.client("dynamodb")

class TaskTable():
    def __init__(self):
        self.taskTable = os.environ["TASK_TABLE"]

    def update_item(self, job_id, key, value):
        dynamodb.update_item(
            TableName = self.taskTable,
            Key = {
                "JobId": { "S": job_id },
            },
            UpdateExpression = "SET #S = :s",
            ExpressionAttributeNames = {
                "#S": key,
            },
            ExpressionAttributeValues = {
                ":s": { "S": value },
            }
        )

    def get_item(self, job_id, key=None):
        response = dynamodb.get_item(
            TableName = self.taskTable,
            Key = {
                "JobId": { "S": job_id },
            }
        )
        item = response.get("Item", None)
        if item is None:
            return None
        if key is None:
            return item
        elif key in item:
            return item[key]["S"]
        else:
            return None

    def put_item(self, job_id, task_type):
        dynamodb.put_item(
            TableName = self.taskTable,
            Item = {
                "JobId": { "S": job_id },
                "TrainingTaskId": { "S": "" },
                "TaskType": { "S": task_type },
                "Status": { "S": "NotStarted" },
                "TrainingDuration": { "S": "0:00:00" },
                "InferenceTaskId": { "S": "" },
            }
        )

    def delete_item(self, job_id):
        dynamodb.delete_item(
            TableName = self.taskTable,
            Key = {
                "JobId": { "S": job_id },
            }
        )

    def scan(self):
        response = dynamodb.scan(TableName=self.taskTable)
        items = response["Items"]
        return items
        
