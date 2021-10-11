# Deploy via AWS CDK

## Prerequisites

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

_Note_: Use `cn-northwest-1` region and `727897471807` account ID currently.

```
// AWS Standard Regions
aws ecr get-login-password --region us-west-2 --profile <profile> | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

// AWS China Regions
aws ecr get-login-password --region <region> --profile <profile> | docker login --username AWS --password-stdin 727897471807.dkr.ecr.<region>.amazonaws.com.cn
```

### Build the Web Portal

```
cd source/portal/
npm install
npm run build
```

### CDK Synth & CDK Deploy
_Note_: Please make sure Docker is running.

#### AWS Commercial Regions
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
| LoginDomain      | Yes      |         | The authentication URL of [authing.cn](https://authing.cn)      |
