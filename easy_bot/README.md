# AWS Machine Learning Bot

AWS Machine Learning Bot (ML Bot) is a solution that allows users without machine learning expertise to train machine 
learning models with their own datasets.

![](website/content/images/homepage.jpg)

## Features

- [x] Authentication
- [x] Self-service User Interface
    - [x] Import Data from S3
    - [x] Start Training Jobs
    - [x] Prediction with Data Samples
- [x] CDK Deployment
- [x] CloudFormation Deployment
- [x] Computer Vision
    - [x] Image Classification
    - [x] Object Detection
- [x] Natural Language Processing
    - [x] Named Entity Recognition

## Architecture

![](website/content/images/architecture.png)

A web portal will be launched in the AWS account. Through the web portal, users can create machine learning tasks and manage them in a centralized portal. 

## Deploy via CloudFormation

   [![Launch Stack](website/content/images/launch-stack.png)](https://console.amazonaws.cn/cloudformation/home#/stacks/create/template?stackName=ml-bot&templateURL=https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/ml-bot/v1.1.2/ml-bot.template) in AWS China Regions (cn-northwest-1, cn-north-1)

   [![Launch Stack](website/content/images/launch-stack.png)](https://us-west-2.console.aws.amazon.com/cloudformation/home#/stacks/create/template?stackName=ml-bot&templateURL=https://aws-gcr-solutions.s3.amazonaws.com/ml-bot/v1.1.2/aws-ml-bot.template) in AWS Standard Regions

## Deploy via AWS CDK

_Note_: You should choose either [Deploy via CloudFormation](#deploy-via-cloudformation) or [Deploy via AWS CDK](#deploy-via-aws-cdk). If you are don't want to install the dependencies on your local machine. Please choose [Deploy via CloudFormation](#deploy-via-cloudformation).

### Prerequisites

Please install the following dependencies on your local machine.

* nodejs 12+
* npm 6+
* Docker
* AWS CLI

You need CDK bootstrap v4+ to deploy this application. To upgrade to latest CDK bootstrap version. Run
```
cdk bootstrap s3://<account-id>/<aws-region>
```

Please make sure Docker is running on your local machine.

### Login into Amazon ECR
The solution uses the [Deep Machine AMIs](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) 
distributed through Amazon ECR. You need to login Amazon ECR before deploy the CDK. 

```
// AWS Standard Regions
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

// AWS China Regions
aws ecr get-login-password --region cn-north-1 | docker login --username AWS --password-stdin 727897471807.dkr.ecr.<region>.amazonaws.com.cn
```

### CDK Synth & CDK Deploy
_Note_: Please make sure Docker is running.

#### AWS Standard Regions
```
cd ../constructs
npm install
npm run build
npx cdk deploy
```

#### AWS China Regions

```
cd ../constructs
npm install 
npm run build
npx cdk deploy --context Partition=aws-cn --parameters Domain=xxx.yy --parameter SSLCertificateId=xxxxxx --parameters LoginDomain=xxx.authing.cn
```

**AWS China Regions Parameters**

| Parameter        | Required | Default | Description                                                  |
| ---------------- | -------- | ------- | ------------------------------------------------------------ |
| Domain           | Yes      |         | The CNAME associated with the web portal. The domain must be ICP licensed.      |
| SSLCertificateId | Yes      |         | The SSL Certificate ID in AWS IAM. You must upload the SSL certificate to AWS IAM using [AWS CLI upload-server-certificate](https://docs.aws.amazon.com/cli/latest/reference/iam/upload-server-certificate.html) command before deploy this solution. The SSL certificate should be under `/cloudfront/` path. |
| LoginDomain      | Yes      |         | The login domain associated with the web portal, e.g. ml-bot-xxx.authing.cn |
