# mlbot解决方案cdk编译方法
<!--BEGIN STABILITY BANNER-->
---

![Stability: Experimental](https://img.shields.io/badge/stability-Experimental-important.svg?style=for-the-badge)

> **This is an experimental example. It may not build out of the box**
>
> This examples does is built on Construct Libraries marked "Experimental" and may not be updated for latest breaking changes.
>
> If build is unsuccessful, please create an [issue](https://github.com/aws-samples/aws-cdk-examples/issues/new) so that we may debug the problem

---
<!--END STABILITY BANNER-->

This example creates the infrastructure for a static site, which uses an S3 bucket for storing the content.  The site contents (located in the 'site-contents' sub-directory) are deployed to the bucket.

The site redirects from HTTP to HTTPS, using a CloudFront distribution, Route53 alias record, and ACM certificate.

## Prep

The domain for the static site (i.e. mystaticsite.com) must be configured as a hosted zone in Route53 prior to deploying this example.  For instructions on configuring Route53 as the DNS service for your domain, see the [Route53 documentation](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/dns-configuring.html).


## First
先安装好以下库，版本必须满足要求。
Please install the following dependencies on your local machine.

* nodejs 12+
```
sudo apt install nodejs
```
* npm 6+
```
sudo apt install npm
```
* Docker
* AWS CLI
```
sudo apt-get install awscli
```

更新npm方式：
```
第一步，先查看本机node.js版本：
node -v
第二步，清除node.js的cache：
sudo npm cache clean -f
第三步，安装Node版本管理工具，工具的名字有点奇葩，叫做“n”
sudo npm install -g n
第四步，安装最新版本的node.js
sudo n stable
第五步，再次查看本机的node.js版本：
node -v
第六步，更新npm到最新版：
sudo npm install npm@latest -g
第七步，验证版本是否升级
node -v
npm -v
```

## Deploy

```
cd MLBot/source/constructs/
sudo npm install -g aws-cdk
npm install
npm run build

查看AWS配置文件中的密钥：
首先进入IAM的我的安全凭证，创建访问密钥，将密钥粘到AWS_ACCESS_KEY_ID和AWS_SECRET_ACCESS_KEY上去。
aws configure list
export AWS_ACCESS_KEY_ID=AKIAU4KE6XOVCJPZ6CRJ
export AWS_SECRET_ACCESS_KEY=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
export AWS_DEFAULT_REGION=cn-northwest-1

编译前端界面
cd ../portal/
npm install
npm run build
报错运行：
sudo chown -R 1000:1000 "/home/edge/.npm"

修改配置中的路径
cd ../constructs/
vim ecs-stack.ts
54: '727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn' :
'727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn' :

bootstrap 已经运行过
cdk bootstrap aws://335688940458/cn-northwest-1 # run this line at the first time

中国区需要先登陆
aws ecr get-login-password --region cn-northwest-1 | docker login --username AWS --password-stdin 727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn
修改域名Domin以及LoginDomain
cdk deploy ml-bot -c EcrOrigin=asset -c Partition=aws-cn  --parameters Domain=aws.airuntime.cn --parameters LoginDomain=ml-bot-workshop-airuntime.authing.cn --require-approval=never
每次编译前删除cdk.out
sudo rm -rf cdk.out
npm run build
```

编译通过build_and_push进行远程更新
```
sudo rm -rf cdk.out/
npm run build
aws ecr get-login-password --region cn-northwest-1 | docker login --username AWS --password-stdin 727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn
cdk deploy ml-bot  -c Partition=aws-cn  --parameters Domain=aws1.singray-ai.com --parameters LoginDomain=ml-bot-workshop-edge.authing.cn --require-approval=never
cd ../docker/training or cd ../docker/inference
bash build_and_push.sh ml-bot-training cn-northwest-1 v1.1.2
```