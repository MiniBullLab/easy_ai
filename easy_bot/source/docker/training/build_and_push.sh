#!/bin/bash
set -v
set -e
# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=${1:-ml-bot-training}

# Get the region defined in the current configuration
region=${2:-cn-north-1}

# Get the tag defined in the current configuration
tag=${3:-$(git describe --tag --exact-match)}

# Get the account number associated with the current IAM credentials
account=${4:-$(aws sts get-caller-identity --query Account --output text)}

if [[ $region =~ ^cn.* ]]
then
    fullname="${account}.dkr.ecr.${region}.amazonaws.com.cn/${image}:${tag}"
    registry_id="727897471807"
    registry_uri="${registry_id}.dkr.ecr.cn-northwest-1.amazonaws.com.cn"
else
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${tag}"
    registry_id="763104351884"
    registry_uri="${registry_id}.dkr.ecr.us-west-2.amazonaws.com"
fi

echo ${fullname}

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image}" --region ${region} || aws ecr create-repository --repository-name "${image}" --region ${region}

# Get the login command from ECR and execute it directly
$(aws ecr get-login --registry-ids ${account} --region ${region} --no-include-email)
$(aws ecr get-login --registry-ids ${registry_id} --region ${region} --no-include-email)

# Build the docker image, tag with full name and then push it to ECR
docker build -t ${image} -f Dockerfile . --build-arg REGISTRY_URI=${registry_uri}
docker tag ${image} ${fullname}
docker push ${fullname}
