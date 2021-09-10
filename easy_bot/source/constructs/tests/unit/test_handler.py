import json
import pytest
from ml_bot import app


@pytest.fixture()
def apigw_event():
    """ Generates API GW Event"""
    return {
        "httpMethod": "POST",
        "resource": {},
        "body": {},
    }

@pytest.fixture()
def sagemaker_event():
    """ Generates SageMaker Event"""
    return {
        "source": "aws.sagemaker",
        "detail": {
            "TrainingJobName": "ml-bot-image-classifition-2020-10-01-00-00-00-000",
            "SecondaryStatusTransitions": [{
                "Status": "Completed",
                "StatusMessage": "",
            },],
            "ModelArtifacts": {
                "S3ModelArtifacts": "",
            },
        },
    }


def test_lambda_handler(apigw_event, mocker):
    ret = app.lambda_handler(apigw_event, "")
    data = json.loads(ret["body"])
    assert ret["statusCode"] == 200

def test_sagemaker_event_handler(sagemaker_event, mocker):
    ret = app.lambda_handler(sagemaker_event, "")
    data = json.loads(ret["body"])
    assert ret["statusCode"] == 200
