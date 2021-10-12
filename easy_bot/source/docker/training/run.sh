
export AWS_DEFAULT_REGION="cn-north-1"
export DATA_BUCKET="ml-bot-lambdaapidatabucket19fa1f65-ow9mxx0sbqe2"
export DATA_PREFIX="named-entity-recognition/ratings-1"
export HYPERPARAMETERS="{'EPOCHS': '80', 'LANGUAGE': 'en'}"
export MODEL_BUCKET="ml-bot-lambdaapimodelbucketf978ec1b-jsl4vx9f8pcs"
export MODEL_PREFIX="named-entity-recognition/ratings-1"
export TASK_FAMILY="Training"
export TASK_NAME="ratings-1"
export TASK_TYPE="NAMED_ENTITY_RECOGNITION"

/home/ubuntu/anaconda3/bin/python3 train.py

