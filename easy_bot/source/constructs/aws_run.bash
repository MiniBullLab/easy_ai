sudo rm -rf cdk.out
npm run build

cdk bootstrap aws://335688940458/cn-northwest-1

aws ecr get-login-password --region cn-northwest-1 | docker login --username AWS --password-stdin 727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn

cdk deploy ml-bot -c EcrOrigin=asset -c Partition=aws-cn  --parameters Domain=aws.cj-zn.com --parameters LoginDomain=ml-bot-workshop-airuntime.authing.cn --require-approval=never