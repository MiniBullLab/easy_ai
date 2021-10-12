"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.addCorsOptions = exports.LambdaApiStack = void 0;
const iam = require("@aws-cdk/aws-iam");
const lambda = require("@aws-cdk/aws-lambda");
const cdk = require("@aws-cdk/core");
const apigateway = require("@aws-cdk/aws-apigateway");
const dynamodb = require("@aws-cdk/aws-dynamodb");
const path = require("path");
const s3 = require("@aws-cdk/aws-s3");
const cognito = require("@aws-cdk/aws-cognito");
const efs = require("@aws-cdk/aws-efs");
const ec2 = require("@aws-cdk/aws-ec2");
const codebuild = require("@aws-cdk/aws-codebuild");
const cr = require("@aws-cdk/custom-resources");
class LambdaApiStack extends cdk.Construct {
    constructor(parent, id, props) {
        super(parent, id);
        this.taskTable = new dynamodb.Table(this, 'TaskTable', {
            partitionKey: { name: 'JobId', type: dynamodb.AttributeType.STRING },
            billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
            removalPolicy: cdk.RemovalPolicy.DESTROY
        });
        // this.vpc = new Vpc(this, 'Vpc', { maxAzs: 2, natGateways: 1 });
        this.dataBucket = new s3.Bucket(this, "DataBucket", {
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: true
        });
        this.modelBucket = new s3.Bucket(this, "ModelBucket", {
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: true
        });
        const prodBucket = new s3.Bucket(this, "ProdBucket", {
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: false
        });
        this.restApi = new apigateway.RestApi(this, "RestApi", {
            restApiName: cdk.Aws.STACK_NAME,
            deployOptions: {
                stageName: "Prod",
                metricsEnabled: true,
                loggingLevel: apigateway.MethodLoggingLevel.INFO,
                dataTraceEnabled: true,
            },
            endpointConfiguration: {
                types: [apigateway.EndpointType.REGIONAL]
            }
        });
        // Security Group definitions.
        this.ec2SecurityGroup = new ec2.SecurityGroup(this, 'LambdaEFSMLEC2SG', {
            vpc: props.vpc, allowAllOutbound: true,
        });
        const lambdaSecurityGroup = new ec2.SecurityGroup(this, 'LambdaEFSMLLambdaSG', {
            vpc: props.vpc, allowAllOutbound: true,
        });
        const efsSecurityGroup = new ec2.SecurityGroup(this, 'LambdaEFSMLEFSSG', {
            vpc: props.vpc, allowAllOutbound: true,
        });
        this.ec2SecurityGroup.connections.allowTo(efsSecurityGroup, ec2.Port.tcp(2049));
        lambdaSecurityGroup.connections.allowTo(efsSecurityGroup, ec2.Port.tcp(2049));
        const s3ReadPolicy = new iam.PolicyStatement();
        s3ReadPolicy.addActions("s3:GetObject*");
        s3ReadPolicy.addActions("s3:ListBucket");
        s3ReadPolicy.addResources("*");
        const s3Policy = new iam.PolicyStatement();
        s3Policy.addActions("s3:*");
        s3Policy.addResources(this.dataBucket.bucketArn + "/*");
        s3Policy.addResources(this.modelBucket.bucketArn + "/*");
        s3Policy.addResources(prodBucket.bucketArn + "/*");
        const s3controlPolicy = new iam.PolicyStatement();
        s3controlPolicy.addActions("s3:*");
        s3controlPolicy.addActions("iam:*");
        s3controlPolicy.addResources("*");
        const ecsPolicy = new iam.PolicyStatement();
        ecsPolicy.addActions("ec2:*");
        ecsPolicy.addActions("ecs:*");
        ecsPolicy.addActions("iam:*");
        ecsPolicy.addResources("*");
        const sagemakerPolicy = new iam.PolicyStatement();
        sagemakerPolicy.addActions("sagemaker:*");
        sagemakerPolicy.addResources("*");
        const dynamodbPolicy = new iam.PolicyStatement();
        dynamodbPolicy.addActions("dynamodb:*");
        dynamodbPolicy.addActions("iam:*");
        dynamodbPolicy.addResources(this.taskTable.tableArn);
        const asgPolicy = new iam.PolicyStatement();
        asgPolicy.addActions("autoscaling:*");
        asgPolicy.addActions("elasticloadbalancing:*");
        asgPolicy.addActions("cloudwatch:*");
        asgPolicy.addResources("*");
        const batchOpsRole = new iam.Role(this, "AdminRoleForS3BatchOperations", {
            assumedBy: new iam.ServicePrincipal('batchoperations.s3.amazonaws.com'),
            description: "Allows S3 Batch Operations to call AWS services on your behalf.",
            managedPolicies: [
                iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonS3FullAccess")
            ]
        });
        const sagemakerExecRole = new iam.Role(this, "AdminRoleForSageMakerExecution", {
            assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
            description: "Allows SageMaker Endpoints to call AWS services on your behalf.",
            managedPolicies: [
                iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonS3FullAccess"),
                iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonEC2ContainerRegistryFullAccess"),
                iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess"),
            ]
        });
        // Elastic File System file system.
        // For the purpose of cost saving, provisioned throughput has been kept low.
        this.fileSystem = new efs.FileSystem(this, 'FileSystem', {
            vpc: props.vpc,
            securityGroup: efsSecurityGroup,
            throughputMode: efs.ThroughputMode.PROVISIONED,
            provisionedThroughputPerSecond: cdk.Size.mebibytes(10),
            removalPolicy: cdk.RemovalPolicy.DESTROY
        });
        this.accessPoint = new efs.AccessPoint(this, 'AccessPoint', {
            fileSystem: this.fileSystem,
            path: '/',
            posixUser: {
                gid: '0',
                uid: '0'
            },
            createAcl: {
                ownerGid: '0',
                ownerUid: '0',
                permissions: '777'
            }
        });
        // Leveraging on AWS CodeBuild to install Python libraries to EFS.
        const codeBuildProject = new codebuild.Project(this, 'LambdaEFSMLCodeBuildProject', {
            projectName: "LambdaEFSMLCodeBuildProject",
            description: "Installs Python libraries to EFS.",
            vpc: props.vpc,
            buildSpec: codebuild.BuildSpec.fromObject({
                version: '0.1',
                phases: {
                    build: {
                        commands: [
                            'mkdir -p /mnt/ml',
                            'mkdir -p /mnt/ml/model',
                            'python3 -m venv /mnt/ml/code',
                            '/mnt/ml/code/bin/pip3 install -i https://opentuna.cn/pypi/web/simple opencv-python==4.4.0.44',
                            '/mnt/ml/code/bin/pip3 install -i https://opentuna.cn/pypi/web/simple gluoncv==0.8.0',
                            '/mnt/ml/code/bin/pip3 install -i https://opentuna.cn/pypi/web/simple gluonnlp==0.10.0',
                            '/mnt/ml/code/bin/pip3 install -i https://opentuna.cn/pypi/web/simple mxnet-mkl==1.6.0',
                            'chown -R 1000:1000 /mnt/ml/'
                        ]
                    }
                },
            }),
            environment: {
                buildImage: codebuild.LinuxBuildImage.fromDockerRegistry('lambci/lambda:build-python3.7'),
                computeType: codebuild.ComputeType.LARGE,
                privileged: true,
            },
            securityGroups: [this.ec2SecurityGroup],
            subnetSelection: props.vpc.selectSubnets({ subnetType: ec2.SubnetType.PRIVATE }),
            timeout: cdk.Duration.minutes(60),
        });
        // Configure EFS for CodeBuild.
        const partition = parent.node.tryGetContext('Partition');
        const cfnProject = codeBuildProject.node.defaultChild;
        cfnProject.fileSystemLocations = [{
                type: "EFS",
                location: (partition === 'aws-cn') ?
                    `${this.fileSystem.fileSystemId}.efs.${cdk.Aws.REGION}.amazonaws.com.cn:/` :
                    `${this.fileSystem.fileSystemId}.efs.${cdk.Aws.REGION}.amazonaws.com:/`,
                mountPoint: "/mnt/ml",
                identifier: "ml",
                mountOptions: "nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2"
            }];
        cfnProject.logsConfig = {
            cloudWatchLogs: {
                status: "ENABLED"
            }
        };
        cfnProject.addPropertyOverride('Environment.ImagePullCredentialsType', 'CODEBUILD');
        // Triggers the CodeBuild project to install the python packages and model to the EFS file system
        const triggerBuildProject = new cr.AwsCustomResource(this, 'TriggerCodeBuild', {
            onCreate: {
                service: 'CodeBuild',
                action: 'startBuild',
                parameters: {
                    projectName: codeBuildProject.projectName
                },
                physicalResourceId: cr.PhysicalResourceId.fromResponse('build.id'),
            },
            onUpdate: {
                service: 'CodeBuild',
                action: 'startBuild',
                parameters: {
                    projectName: codeBuildProject.projectName
                },
                physicalResourceId: cr.PhysicalResourceId.fromResponse('build.id'),
            },
            policy: cr.AwsCustomResourcePolicy.fromSdkCalls({ resources: cr.AwsCustomResourcePolicy.ANY_RESOURCE })
        });
        // Create dependency between EFS and Codebuild
        codeBuildProject.node.addDependency(this.accessPoint);
        this.lambdaFunction = new lambda.Function(this, "LambdaFunction", {
            handler: "app.lambda_handler",
            runtime: lambda.Runtime.PYTHON_3_7,
            code: lambda.Code.fromAsset(path.join(__dirname, './src/'), {
                bundling: {
                    image: lambda.Runtime.PYTHON_3_7.bundlingDockerImage,
                    command: [
                        'bash', '-c', [
                            `cp -r /asset-input/* /asset-output/`,
                            `cd /asset-output/`,
                            `chmod a+x ./setup.sh`,
                            `./setup.sh`, `ls -lah`
                        ].join(' && ')
                    ],
                    user: 'root'
                }
            }),
            memorySize: 2048,
            timeout: cdk.Duration.seconds(30),
            environment: {
                DATA_BUCKET: this.dataBucket.bucketName,
                MODEL_BUCKET: this.modelBucket.bucketName,
                PROD_BUCKET: prodBucket.bucketName,
                TASK_TABLE: this.taskTable.tableName,
                BATCH_OPS_ROLE_ARN: batchOpsRole.roleArn,
                SAGEMAKER_EXEC_ROLE_ARN: sagemakerExecRole.roleArn,
                PRIVATE_SUBNETS: props.privateSubnets,
                VPC_ID: props.vpc.vpcId,
            },
            initialPolicy: [
                s3ReadPolicy, s3Policy, s3controlPolicy, ecsPolicy, sagemakerPolicy, dynamodbPolicy, asgPolicy
            ],
            securityGroup: lambdaSecurityGroup,
            vpc: props.vpc,
            filesystem: lambda.FileSystem.fromEfsAccessPoint(this.accessPoint, "/mnt/ml"),
        });
        this.dataBucket.grantReadWrite(this.lambdaFunction);
        this.modelBucket.grantReadWrite(this.lambdaFunction);
        prodBucket.grantReadWrite(this.lambdaFunction);
        if (props.authType === "cognito" /* COGNITO */) {
            // Create Cognito User Pool
            this.userPool = new cognito.UserPool(this, 'UserPool', {
                selfSignUpEnabled: false,
                signInCaseSensitive: false,
                signInAliases: {
                    email: true,
                    username: false,
                    phone: true
                }
            });
            // Create User Pool Client
            this.userPoolApiClient = new cognito.UserPoolClient(this, 'UserPoolApiClient', {
                userPool: this.userPool,
                userPoolClientName: 'ReplicationHubPortal',
                preventUserExistenceErrors: true
            });
        }
        const lambdaFn = new apigateway.LambdaIntegration(this.lambdaFunction, {});
        const customOptions = {
            authorizationType: apigateway.AuthorizationType.CUSTOM,
            authorizer: new apigateway.TokenAuthorizer(this, "TokenAuthorizer", {
                handler: this.lambdaFunction,
                identitySource: "method.request.header.authorization",
                resultsCacheTtl: cdk.Duration.minutes(5)
            })
        };
        const methodOptions = customOptions;
        const taskApi = this.restApi.root.addResource('tasks');
        taskApi.addMethod('GET', lambdaFn, methodOptions);
        taskApi.addMethod('POST', lambdaFn, methodOptions);
        const taskIdApi = taskApi.addResource('{task_id}');
        taskIdApi.addMethod('DELETE', lambdaFn, methodOptions);
        taskIdApi.addMethod('GET', lambdaFn, methodOptions);
        const taskStopApi = taskIdApi.addResource('stop');
        taskStopApi.addMethod('POST', lambdaFn, methodOptions);
        const taskDataApi = taskIdApi.addResource('data');
        taskDataApi.addMethod('GET', lambdaFn, methodOptions);
        const taskStatusApi = taskIdApi.addResource('status');
        taskStatusApi.addMethod('GET', lambdaFn, methodOptions);
        const taskDataClassIdApi = taskDataApi.addResource('{class_id}');
        taskDataClassIdApi.addMethod('GET', lambdaFn, methodOptions);
        taskDataClassIdApi.addMethod('POST', lambdaFn, methodOptions);
        const taskS3DataApi = taskIdApi.addResource('s3data');
        taskS3DataApi.addMethod('GET', lambdaFn, methodOptions);
        taskS3DataApi.addMethod('POST', lambdaFn, methodOptions);
        const taskTrainApi = taskIdApi.addResource('train');
        taskTrainApi.addMethod('POST', lambdaFn, methodOptions);
        const taskPredictApi = taskIdApi.addResource('predict');
        taskPredictApi.addMethod('POST', lambdaFn, methodOptions);
        const taskPredictV2Api = taskIdApi.addResource('predict_v2');
        taskPredictV2Api.addMethod('POST', lambdaFn, methodOptions);
        const taskDeployApi = taskIdApi.addResource('deploy');
        taskDeployApi.addMethod('POST', lambdaFn, methodOptions);
        addCorsOptions(taskApi);
        addCorsOptions(taskIdApi);
        addCorsOptions(taskStopApi);
        addCorsOptions(taskStatusApi);
        addCorsOptions(taskDataApi);
        addCorsOptions(taskDataClassIdApi);
        addCorsOptions(taskS3DataApi);
        addCorsOptions(taskTrainApi);
        addCorsOptions(taskPredictApi);
        addCorsOptions(taskPredictV2Api);
        addCorsOptions(taskDeployApi);
    }
}
exports.LambdaApiStack = LambdaApiStack;
function addCorsOptions(apiResource) {
    apiResource.addMethod('OPTIONS', new apigateway.MockIntegration({
        integrationResponses: [{
                statusCode: '200',
                responseParameters: {
                    'method.response.header.Access-Control-Allow-Headers': "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,X-Amz-User-Agent'",
                    'method.response.header.Access-Control-Allow-Origin': "'*'",
                    'method.response.header.Access-Control-Allow-Credentials': "'false'",
                    'method.response.header.Access-Control-Allow-Methods': "'OPTIONS,GET,PUT,POST,DELETE'",
                },
            }],
        passthroughBehavior: apigateway.PassthroughBehavior.NEVER,
        requestTemplates: {
            "application/json": "{\"statusCode\": 200}"
        },
    }), {
        methodResponses: [{
                statusCode: '200',
                responseParameters: {
                    'method.response.header.Access-Control-Allow-Headers': true,
                    'method.response.header.Access-Control-Allow-Methods': true,
                    'method.response.header.Access-Control-Allow-Credentials': true,
                    'method.response.header.Access-Control-Allow-Origin': true,
                },
            }],
    });
}
exports.addCorsOptions = addCorsOptions;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibGFtYmRhLWFwaS1zdGFjay5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbImxhbWJkYS1hcGktc3RhY2sudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7O0FBQUEsd0NBQXVDO0FBQ3ZDLDhDQUE4QztBQUU5QyxxQ0FBcUM7QUFDckMsc0RBQXNEO0FBQ3RELGtEQUFrRDtBQUNsRCw2QkFBNkI7QUFDN0Isc0NBQXNDO0FBQ3RDLGdEQUFnRDtBQUNoRCx3Q0FBd0M7QUFDeEMsd0NBQXdDO0FBQ3hDLG9EQUFtRDtBQUNuRCxnREFBK0M7QUFTL0MsTUFBYSxjQUFlLFNBQVEsR0FBRyxDQUFDLFNBQVM7SUFnQi9DLFlBQVksTUFBcUIsRUFBRSxFQUFVLEVBQUUsS0FBMEI7UUFDdkUsS0FBSyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsQ0FBQTtRQUVqQixJQUFJLENBQUMsU0FBUyxHQUFHLElBQUksUUFBUSxDQUFDLEtBQUssQ0FBQyxJQUFJLEVBQUUsV0FBVyxFQUFFO1lBQ3JELFlBQVksRUFBRSxFQUFFLElBQUksRUFBRSxPQUFPLEVBQUUsSUFBSSxFQUFFLFFBQVEsQ0FBQyxhQUFhLENBQUMsTUFBTSxFQUFFO1lBQ3BFLFdBQVcsRUFBRSxRQUFRLENBQUMsV0FBVyxDQUFDLGVBQWU7WUFDakQsYUFBYSxFQUFFLEdBQUcsQ0FBQyxhQUFhLENBQUMsT0FBTztTQUN6QyxDQUFDLENBQUM7UUFFSCxrRUFBa0U7UUFDbEUsSUFBSSxDQUFDLFVBQVUsR0FBRyxJQUFJLEVBQUUsQ0FBQyxNQUFNLENBQUMsSUFBSSxFQUFFLFlBQVksRUFBRTtZQUNsRCxhQUFhLEVBQUUsR0FBRyxDQUFDLGFBQWEsQ0FBQyxPQUFPO1lBQ3hDLGlCQUFpQixFQUFFLElBQUk7U0FDeEIsQ0FBQyxDQUFBO1FBQ0YsSUFBSSxDQUFDLFdBQVcsR0FBRyxJQUFJLEVBQUUsQ0FBQyxNQUFNLENBQUMsSUFBSSxFQUFFLGFBQWEsRUFBRTtZQUNwRCxhQUFhLEVBQUUsR0FBRyxDQUFDLGFB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QzlELG9CQUFvQixFQUFFLENBQUM7Z0JBQ3JCLFVBQVUsRUFBRSxLQUFLO2dCQUNqQixrQkFBa0IsRUFBRTtvQkFDbEIscURBQXFELEVBQUUseUZBQXlGO29CQUNoSixvREFBb0QsRUFBRSxLQUFLO29CQUMzRCx5REFBeUQsRUFBRSxTQUFTO29CQUNwRSxxREFBcUQsRUFBRSwrQkFBK0I7aUJBQ3ZGO2FBQ0YsQ0FBQztRQUNGLG1CQUFtQixFQUFFLFVBQVUsQ0FBQyxtQkFBbUIsQ0FBQyxLQUFLO1FBQ3pELGdCQUFnQixFQUFFO1lBQ2hCLGtCQUFrQixFQUFFLHVCQUF1QjtTQUM1QztLQUNGLENBQUMsRUFBRTtRQUNGLGVBQWUsRUFBRSxDQUFDO2dCQUNoQixVQUFVLEVBQUUsS0FBSztnQkFDakIsa0JBQWtCLEVBQUU7b0JBQ2xCLHFEQUFxRCxFQUFFLElBQUk7b0JBQzNELHFEQUFxRCxFQUFFLElBQUk7b0JBQzNELHlEQUF5RCxFQUFFLElBQUk7b0JBQy9ELG9EQUFvRCxFQUFFLElBQUk7aUJBQzNEO2FBQ0YsQ0FBQztLQUNILENBQUMsQ0FBQTtBQUNKLENBQUM7QUExQkQsd0NBMEJDIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgaWFtIGZyb20gXCJAYXdzLWNkay9hd3MtaWFtXCJcbmltcG9ydCAqIGFzIGxhbWJkYSBmcm9tIFwiQGF3cy1jZGsvYXdzLWxhbWJkYVwiO1xuaW1wb3J0IHtWcGN9IGZyb20gXCJAYXdzLWNkay9hd3MtZWMyXCJcbmltcG9ydCAqIGFzIGNkayBmcm9tIFwiQGF3cy1jZGsvY29yZVwiO1xuaW1wb3J0ICogYXMgYXBpZ2F0ZXdheSBmcm9tICdAYXdzLWNkay9hd3MtYXBpZ2F0ZXdheSc7XG5pbXBvcnQgKiBhcyBkeW5hbW9kYiBmcm9tICdAYXdzLWNkay9hd3MtZHluYW1vZGInO1xuaW1wb3J0ICogYXMgcGF0aCBmcm9tICdwYXRoJztcbmltcG9ydCAqIGFzIHMzIGZyb20gJ0Bhd3MtY2RrL2F3cy1zMyc7XG5pbXBvcnQgKiBhcyBjb2duaXRvIGZyb20gJ0Bhd3MtY2RrL2F3cy1jb2duaXRvJztcbmltcG9ydCAqIGFzIGVmcyBmcm9tICdAYXdzLWNkay9hd3MtZWZzJztcbmltcG9ydCAqIGFzIGVjMiBmcm9tICdAYXdzLWNkay9hd3MtZWMyJztcbmltcG9ydCAqIGFzIGNvZGVidWlsZCBmcm9tIFwiQGF3cy1jZGsvYXdzLWNvZGVidWlsZFwiXG5pbXBvcnQgKiBhcyBjciBmcm9tIFwiQGF3cy1jZGsvY3VzdG9tLXJlc291cmNlc1wiXG5pbXBvcnQgeyBBdXRoVHlwZSB9IGZyb20gJy4vaW5kZXgnO1xuXG5leHBvcnQgaW50ZXJmYWNlIExhbWJkYUFwaVN0YWNrUHJvcHMge1xuICByZWFkb25seSBhdXRoVHlwZTogQXV0aFR5cGU7XG4gIHJlYWRvbmx5IHZwYzogZWMyLklWcGM7XG4gIHJlYWRvbmx5IHByaXZhdGVTdWJuZXRzOiBzdHJpbmc7XG59XG5cbmV4cG9ydCBjbGFzcyBMYW1iZGFBcGlTdGFjayBleHRlbmRzIGNkay5Db25zdHJ1Y3Qge1xuICByZWFkb25seSByZXN0QXBpOiBhcGlnYXRld2F5LlJlc3RBcGlcbiAgcmVhZG9ubHkgbGFtYmRhRnVuY3Rpb246IGxhbWJkYS5GdW5jdGlvblxuICByZWFkb25seSBkYXRhQnVja2V0OiBzMy5CdWNrZXRcbiAgcmVhZG9ubHkgbW9kZWxCdWNrZXQ6IHMzLkJ1Y2tldFxuICAvLyByZWFkb25seSB2cGM6IFZwY1xuICByZWFkb25seSB0YXNrVGFibGU6IGR5bmFtb2RiLlRhYmxlXG5cbiAgcmVhZG9ubHkgdXNlclBvb2w/OiBjb2duaXRvLlVzZXJQb29sXG4gIHJlYWRvbmx5IHVzZXJQb29sQXBpQ2xpZW50PzogY29nbml0by5Vc2VyUG9vbENsaWVudFxuICByZWFkb25seSB1c2VyUG9vbERvbWFpbj86IGNvZ25pdG8uVXNlclBvb2xEb21haW5cblxuICByZWFkb25seSBmaWxlU3lzdGVtIDogZWZzLkZpbGVTeXN0ZW1cbiAgcmVhZG9ubHkgYWNjZXNzUG9pbnQgOiBlZnMuQWNjZXNzUG9pbnRcbiAgcmVhZG9ubHkgZWMyU2VjdXJpdHlHcm91cCA6IGVjMi5TZWN1cml0eUdyb3VwXG5cbiAgY29uc3RydWN0b3IocGFyZW50OiBjZGsuQ29uc3RydWN0LCBpZDogc3RyaW5nLCBwcm9wczogTGFtYmRhQXBpU3RhY2tQcm9wcykge1xuICAgIHN1cGVyKHBhcmVudCwgaWQpXG5cbiAgICB0aGlzLnRhc2tUYWJsZSA9IG5ldyBkeW5hbW9kYi5UYWJsZSh0aGlzLCAnVGFza1RhYmxlJywge1xuICAgICAgcGFydGl0aW9uS2V5OiB7IG5hbWU6ICdKb2JJZCcsIHR5cGU6IGR5bmFtb2RiLkF0dHJpYnV0ZVR5cGUuU1RSSU5HIH0sXG4gICAgICBiaWxsaW5nTW9kZTogZHluYW1vZGIuQmlsbGluZ01vZGUuUEFZX1BFUl9SRVFVRVNULFxuICAgICAgcmVtb3ZhbFBvbGljeTogY2RrLlJlbW92YWxQb2xpY3kuREVTVFJPWVxuICAgIH0pO1xuICAgIFxuICAgIC8vIHRoaXMudnBjID0gbmV3IFZwYyh0aGlzLCAnVnBjJywgeyBtYXhBenM6IDIsIG5hdEdhdGV3YXlzOiAxIH0pO1xuICAgIHRoaXMuZGF0YUJ1Y2tldCA9IG5ldyBzMy5CdWNrZXQodGhpcywgXCJEYXRhQnVja2V0XCIsIHtcbiAgICAgIHJlbW92YWxQb2xpY3k6IGNkay5SZW1vdmFsUG9saWN5LkRFU1RST1ksXG4gICAgICBhdXRvRGVsZXRlT2JqZWN0czogdHJ1ZVxuICAgIH0pXG4gICAgdGhpcy5tb2RlbEJ1Y2tldCA9IG5ldyBzMy5CdWNrZXQodGhpcywgXCJNb2RlbEJ1Y2tldFwiLCB7XG4gICAgICByZW1vdmFsUG9saWN5OiBjZGsuUmVtb3ZhbFBvbGljeS5ERVNUUk9ZLFxuICAgICAgYXV0b0RlbGV0ZU9iamVjdHM6IHRydWVcbiAgICB9KVxuICAgIGNvbnN0IHByb2RCdWNrZXQgPSBuZXcgczMuQnVja2V0KHRoaXMsIFwiUHJvZEJ1Y2tldFwiLCB7XG4gICAgICByZW1vdmFsUG9saWN5OiBjZGsuUmVtb3ZhbFBvbGljeS5ERVNUUk9ZLFxuICAgICAgYXV0b0RlbGV0ZU9iamVjdHM6IGZhbHNlXG4gICAgfSlcblxuICAgIHRoaXMucmVzdEFwaSA9IG5ldyBhcGlnYXRld2F5LlJlc3RBcGkodGhpcywgXCJSZXN0QXBpXCIsIHtcbiAgICAgIHJlc3RBcGlOYW1lOiBjZGsuQXdzLlNUQUNLX05BTUUsXG4gICAgICBkZXBsb3lPcHRpb25zOiB7XG4gICAgICAgIHN0YWdlTmFtZTogXCJQcm9kXCIsXG4gICAgICAgIG1ldHJpY3NFbmFibGVkOiB0cnVlLFxuICAgICAgICBsb2dnaW5nTGV2ZWw6IGFwaWdhdGV3YXkuTWV0aG9kTG9nZ2luZ0xldmVsLklORk8sXG4gICAgICAgIGRhdGFUcmFjZUVuYWJsZWQ6IHRydWUsXG4gICAgICB9LFxuICAgICAgZW5kcG9pbnRDb25maWd1cmF0aW9uOiB7XG4gICAgICAgIHR5cGVzOiBbIGFwaWdhdGV3YXkuRW5kcG9pbnRUeXBlLlJFR0lPTkFMIF1cbiAgICAgIH1cbiAgICB9KVxuXG4gICAgLy8gU2VjdXJpdHkgR3JvdXAgZGVmaW5pdGlvbnMuXG4gICAgdGhpcy5lYzJTZWN1cml0eUdyb3VwID0gbmV3IGVjMi5TZWN1cml0eUdyb3VwKHRoaXMsICdMYW1iZGFFRlNNTEVDMlNHJywge1xuICAgICAgdnBjOiBwcm9wcy52cGMsIGFsbG93QWxsT3V0Ym91bmQ6IHRydWUsXG4gICAgfSk7XG4gICAgY29uc3QgbGFtYmRhU2VjdXJpdHlHcm91cCA9IG5ldyBlYzIuU2VjdXJpdHlHcm91cCh0aGlzLCAnTGFtYmRhRUZTTUxMYW1iZGFTRycsIHtcbiAgICAgIHZwYzogcHJvcHMudnBjLCBhbGxvd0FsbE91dGJvdW5kOiB0cnVlLFxuICAgIH0pO1xuICAgIGNvbnN0IGVmc1NlY3VyaXR5R3JvdXAgPSBuZXcgZWMyLlNlY3VyaXR5R3JvdXAodGhpcywgJ0xhbWJkYUVGU01MRUZTU0cnLCB7XG4gICAgICB2cGM6IHByb3BzLnZwYywgYWxsb3dBbGxPdXRib3VuZDogdHJ1ZSxcbiAgICB9KTtcbiAgICB0aGlzLmVjMlNlY3VyaXR5R3JvdXAuY29ubmVjdGlvbnMuYWxsb3dUbyhlZnNTZWN1cml0eUdyb3VwLCBlYzIuUG9ydC50Y3AoMjA0OSkpO1xuICAgIGxhbWJkYVNlY3VyaXR5R3JvdXAuY29ubmVjdGlvbnMuYWxsb3dUbyhlZnNTZWN1cml0eUdyb3VwLCBlYzIuUG9ydC50Y3AoMjA0OSkpO1xuICAgIFxuICAgIGNvbnN0IHMzUmVhZFBvbGljeSA9IG5ldyBpYW0uUG9saWN5U3RhdGVtZW50KClcbiAgICBzM1JlYWRQb2xpY3kuYWRkQWN0aW9ucyhcInMzOkdldE9iamVjdCpcIilcbiAgICBzM1JlYWRQb2xpY3kuYWRkQWN0aW9ucyhcInMzOkxpc3RCdWNrZXRcIilcbiAgICBzM1JlYWRQb2xpY3kuYWRkUmVzb3VyY2VzKFwiKlwiKVxuICAgIGNvbnN0IHMzUG9saWN5ID0gbmV3IGlhbS5Qb2xpY3lTdGF0ZW1lbnQoKVxuICAgIHMzUG9saWN5LmFkZEFjdGlvbnMoXCJzMzoqXCIpXG4gICAgczNQb2xpY3kuYWRkUmVzb3VyY2VzKHRoaXMuZGF0YUJ1Y2tldC5idWNrZXRBcm4gKyBcIi8qXCIpXG4gICAgczNQb2xpY3kuYWRkUmVzb3VyY2VzKHRoaXMubW9kZWxCdWNrZXQuYnVja2V0QXJuICsgXCIvKlwiKVxuICAgIHMzUG9saWN5LmFkZFJlc291cmNlcyhwcm9kQnVja2V0LmJ1Y2tldEFybiArIFwiLypcIilcbiAgICBjb25zdCBzM2NvbnRyb2xQb2xpY3kgPSBuZXcgaWFtLlBvbGljeVN0YXRlbWVudCgpXG4gICAgczNjb250cm9sUG9saWN5LmFkZEFjdGlvbnMoXCJzMzoqXCIpXG4gICAgczNjb250cm9sUG9saWN5LmFkZEFjdGlvbnMoXCJpYW06KlwiKVxuICAgIHMzY29udHJvbFBvbGljeS5hZGRSZXNvdXJjZXMoXCIqXCIpXG4gICAgY29uc3QgZWNzUG9saWN5ID0gbmV3IGlhbS5Qb2xpY3lTdGF0ZW1lbnQoKVxuICAgIGVjc1BvbGljeS5hZGRBY3Rpb25zKFwiZWMyOipcIilcbiAgICBlY3NQb2xpY3kuYWRkQWN0aW9ucyhcImVjczoqXCIpXG4gICAgZWNzUG9saWN5LmFkZEFjdGlvbnMoXCJpYW06KlwiKVxuICAgIGVjc1BvbGljeS5hZGRSZXNvdXJjZXMoXCIqXCIpXG4gICAgY29uc3Qgc2FnZW1ha2VyUG9saWN5ID0gbmV3IGlhbS5Qb2xpY3lTdGF0ZW1lbnQoKVxuICAgIHNhZ2VtYWtlclBvbGljeS5hZGRBY3Rpb25zKFwic2FnZW1ha2VyOipcIilcbiAgICBzYWdlbWFrZXJQb2xpY3kuYWRkUmVzb3VyY2VzKFwiKlwiKVxuICAgIGNvbnN0IGR5bmFtb2RiUG9saWN5ID0gbmV3IGlhbS5Qb2xpY3lTdGF0ZW1lbnQoKVxuICAgIGR5bmFtb2RiUG9saWN5LmFkZEFjdGlvbnMoXCJkeW5hbW9kYjoqXCIpXG4gICAgZHluYW1vZGJQb2xpY3kuYWRkQWN0aW9ucyhcImlhbToqXCIpXG4gICAgZHluYW1vZGJQb2xpY3kuYWRkUmVzb3VyY2VzKHRoaXMudGFza1RhYmxlLnRhYmxlQXJuKVxuICAgIGNvbnN0IGFzZ1BvbGljeSA9IG5ldyBpYW0uUG9saWN5U3RhdGVtZW50KClcbiAgICBhc2dQb2xpY3kuYWRkQWN0aW9ucyhcImF1dG9zY2FsaW5nOipcIilcbiAgICBhc2dQb2xpY3kuYWRkQWN0aW9ucyhcImVsYXN0aWNsb2FkYmFsYW5jaW5nOipcIilcbiAgICBhc2dQb2xpY3kuYWRkQWN0aW9ucyhcImNsb3Vkd2F0Y2g6KlwiKVxuICAgIGFzZ1BvbGljeS5hZGRSZXNvdXJjZXMoXCIqXCIpXG5cbiAgICBjb25zdCBiYXRjaE9wc1JvbGUgPSBuZXcgaWFtLlJvbGUodGhpcywgXCJBZG1pblJvbGVGb3JTM0JhdGNoT3BlcmF0aW9uc1wiLCB7XG4gICAgICBhc3N1bWVkQnk6IG5ldyBpYW0uU2VydmljZVByaW5jaXBhbCgnYmF0Y2hvcGVyYXRpb25zLnMzLmFtYXpvbmF3cy5jb20nKSxcbiAgICAgIGRlc2NyaXB0aW9uOiBcIkFsbG93cyBTMyBCYXRjaCBPcGVyYXRpb25zIHRvIGNhbGwgQVdTIHNlcnZpY2VzIG9uIHlvdXIgYmVoYWxmLlwiLFxuICAgICAgbWFuYWdlZFBvbGljaWVzOiBbXG4gICAgICAgIGlhbS5NYW5hZ2VkUG9saWN5LmZyb21Bd3NNYW5hZ2VkUG9saWN5TmFtZShcIkFtYXpvblMzRnVsbEFjY2Vzc1wiKVxuICAgICAgXVxuICAgIH0pXG5cbiAgICBjb25zdCBzYWdlbWFrZXJFeGVjUm9sZSA9IG5ldyBpYW0uUm9sZSh0aGlzLCBcIkFkbWluUm9sZUZvclNhZ2VNYWtlckV4ZWN1dGlvblwiLCB7XG4gICAgICBhc3N1bWVkQnk6IG5ldyBpYW0uU2VydmljZVByaW5jaXBhbCgnc2FnZW1ha2VyLmFtYXpvbmF3cy5jb20nKSxcbiAgICAgIGRlc2NyaXB0aW9uOiBcIkFsbG93cyBTYWdlTWFrZXIgRW5kcG9pbnRzIHRvIGNhbGwgQVdTIHNlcnZpY2VzIG9uIHlvdXIgYmVoYWxmLlwiLFxuICAgICAgbWFuYWdlZFBvbGljaWVzOiBbXG4gICAgICAgIGlhbS5NYW5hZ2VkUG9saWN5LmZyb21Bd3NNYW5hZ2VkUG9saWN5TmFtZShcIkFtYXpvblMzRnVsbEFjY2Vzc1wiKSxcbiAgICAgICAgaWFtLk1hbmFnZWRQb2xpY3kuZnJvbUF3c01hbmFnZWRQb2xpY3lOYW1lKFwiQW1hem9uRUMyQ29udGFpbmVyUmVnaXN0cnlGdWxsQWNjZXNzXCIpLFxuICAgICAgICBpYW0uTWFuYWdlZFBvbGljeS5mcm9tQXdzTWFuYWdlZFBvbGljeU5hbWUoXCJBbWF6b25TYWdlTWFrZXJGdWxsQWNjZXNzXCIpLFxuICAgICAgXVxuICAgIH0pXG5cbiAgICAvLyBFbGFzdGljIEZpbGUgU3lzdGVtIGZpbGUgc3lzdGVtLlxuICAgIC8vIEZvciB0aGUgcHVycG9zZSBvZiBjb3N0IHNhdmluZywgcHJvdmlzaW9uZWQgdGhyb3VnaHB1dCBoYXMgYmVlbiBrZXB0IGxvdy5cbiAgICB0aGlzLmZpbGVTeXN0ZW0gPSBuZXcgZWZzLkZpbGVTeXN0ZW0odGhpcywgJ0ZpbGVTeXN0ZW0nLCB7XG4gICAgICB2cGM6IHByb3BzLnZwYyxcbiAgICAgIHNlY3VyaXR5R3JvdXA6IGVmc1NlY3VyaXR5R3JvdXAsXG4gICAgICB0aHJvdWdocHV0TW9kZTogZWZzLlRocm91Z2hwdXRNb2RlLlBST1ZJU0lPTkVELFxuICAgICAgcHJvdmlzaW9uZWRUaHJvdWdocHV0UGVyU2Vjb25kOiBjZGsuU2l6ZS5tZWJpYnl0ZXMoMTApLFxuICAgICAgcmVtb3ZhbFBvbGljeTogY2RrLlJlbW92YWxQb2xpY3kuREVTVFJPWVxuICAgIH0pO1xuICAgIHRoaXMuYWNjZXNzUG9pbnQgPSBuZXcgZWZzLkFjY2Vzc1BvaW50KHRoaXMsICdBY2Nlc3NQb2ludCcsIHtcbiAgICAgIGZpbGVTeXN0ZW06IHRoaXMuZmlsZVN5c3RlbSxcbiAgICAgIHBhdGg6ICcvJywgICAgICAgICAgICAvLyByZW1vdmUgRUZTIGFjY2VzcyBwb2ludFxuICAgICAgcG9zaXhVc2VyOiB7XG4gICAgICAgIGdpZDogJzAnLFxuICAgICAgICB1aWQ6ICcwJ1xuICAgICAgfSxcbiAgICAgIGNyZWF0ZUFjbDoge1xuICAgICAgICBvd25lckdpZDogJzAnLFxuICAgICAgICBvd25lclVpZDogJzAnLFxuICAgICAgICBwZXJtaXNzaW9uczogJzc3NydcbiAgICAgIH1cbiAgICB9KVxuXG4gICAgLy8gTGV2ZXJhZ2luZyBvbiBBV1MgQ29kZUJ1aWxkIHRvIGluc3RhbGwgUHl0aG9uIGxpYnJhcmllcyB0byBFRlMuXG4gICAgY29uc3QgY29kZUJ1aWxkUHJvamVjdCA9IG5ldyBjb2RlYnVpbGQuUHJvamVjdCh0aGlzLCAnTGFtYmRhRUZTTUxDb2RlQnVpbGRQcm9qZWN0Jywge1xuICAgICAgcHJvamVjdE5hbWU6IFwiTGFtYmRhRUZTTUxDb2RlQnVpbGRQcm9qZWN0XCIsXG4gICAgICBkZXNjcmlwdGlvbjogXCJJbnN0YWxscyBQeXRob24gbGlicmFyaWVzIHRvIEVGUy5cIixcbiAgICAgIHZwYzogcHJvcHMudnBjLFxuICAgICAgYnVpbGRTcGVjOiBjb2RlYnVpbGQuQnVpbGRTcGVjLmZyb21PYmplY3Qoe1xuICAgICAgICB2ZXJzaW9uOiAnMC4xJyxcbiAgICAgICAgcGhhc2VzOiB7XG4gICAgICAgICAgYnVpbGQ6IHtcbiAgICAgICAgICAgIGNvbW1hbmRzOiBbXG4gICAgICAgICAgICAgICdta2RpciAtcCAvbW50L21sJyxcbiAgICAgICAgICAgICAgJ21rZGlyIC1wIC9tbnQvbWwvbW9kZWwnLFxuICAgICAgICAgICAgICAncHl0aG9uMyAtbSB2ZW52IC9tbnQvbWwvY29kZScsXG4gICAgICAgICAgICAgICcvbW50L21sL2NvZGUvYmluL3BpcDMgaW5zdGFsbCAtaSBodHRwczovL29wZW50dW5hLmNuL3B5cGkvd2ViL3NpbXBsZSBvcGVuY3YtcHl0aG9uPT00LjQuMC40NCcsXG4gICAgICAgICAgICAgICcvbW50L21sL2NvZGUvYmluL3BpcDMgaW5zdGFsbCAtaSBodHRwczovL29wZW50dW5hLmNuL3B5cGkvd2ViL3NpbXBsZSBnbHVvbmN2PT0wLjguMCcsXG4gICAgICAgICAgICAgICcvbW50L21sL2NvZGUvYmluL3BpcDMgaW5zdGFsbCAtaSBodHRwczovL29wZW50dW5hLmNuL3B5cGkvd2ViL3NpbXBsZSBnbHVvbm5scD09MC4xMC4wJyxcbiAgICAgICAgICAgICAgJy9tbnQvbWwvY29kZS9iaW4vcGlwMyBpbnN0YWxsIC1pIGh0dHBzOi8vb3BlbnR1bmEuY24vcHlwaS93ZWIvc2ltcGxlIG14bmV0LW1rbD09MS42LjAnLFxuICAgICAgICAgICAgICAnY2hvd24gLVIgMTAwMDoxMDAwIC9tbnQvbWwvJ1xuICAgICAgICAgICAgXVxuICAgICAgICAgIH1cbiAgICAgICAgfSxcbiAgICAgIH0pLFxuICAgICAgZW52aXJvbm1lbnQ6IHtcbiAgICAgICAgYnVpbGRJbWFnZTogY29kZWJ1aWxkLkxpbnV4QnVpbGRJbWFnZS5mcm9tRG9ja2VyUmVnaXN0cnkoJ2xhbWJjaS9sYW1iZGE6YnVpbGQtcHl0aG9uMy43JyksXG4gICAgICAgIGNvbXB1dGVUeXBlOiBjb2RlYnVpbGQuQ29tcHV0ZVR5cGUuTEFSR0UsXG4gICAgICAgIHByaXZpbGVnZWQ6IHRydWUsXG4gICAgICB9LFxuICAgICAgc2VjdXJpdHlHcm91cHM6IFt0aGlzLmVjMlNlY3VyaXR5R3JvdXBdLFxuICAgICAgc3VibmV0U2VsZWN0aW9uOiBwcm9wcy52cGMuc2VsZWN0U3VibmV0cyh7IHN1Ym5ldFR5cGU6IGVjMi5TdWJuZXRUeXBlLlBSSVZBVEUgfSksXG4gICAgICB0aW1lb3V0OiBjZGsuRHVyYXRpb24ubWludXRlcyg2MCksXG4gICAgfSk7XG5cbiAgICAvLyBDb25maWd1cmUgRUZTIGZvciBDb2RlQnVpbGQuXG4gICAgY29uc3QgcGFydGl0aW9uID0gcGFyZW50Lm5vZGUudHJ5R2V0Q29udGV4dCgnUGFydGl0aW9uJyk7XG4gICAgY29uc3QgY2ZuUHJvamVjdCA9IGNvZGVCdWlsZFByb2plY3Qubm9kZS5kZWZhdWx0Q2hpbGQgYXMgY29kZWJ1aWxkLkNmblByb2plY3Q7XG4gICAgY2ZuUHJvamVjdC5maWxlU3lzdGVtTG9jYXRpb25zID0gW3tcbiAgICAgIHR5cGU6IFwiRUZTXCIsXG4gICAgICBsb2NhdGlvbjogKHBhcnRpdGlvbiA9PT0gJ2F3cy1jbicpID9cbiAgICAgICAgYCR7dGhpcy5maWxlU3lzdGVtLmZpbGVTeXN0ZW1JZH0uZWZzLiR7Y2RrLkF3cy5SRUdJT059LmFtYXpvbmF3cy5jb20uY246L2AgOlxuICAgICAgICBgJHt0aGlzLmZpbGVTeXN0ZW0uZmlsZVN5c3RlbUlkfS5lZnMuJHtjZGsuQXdzLlJFR0lPTn0uYW1hem9uYXdzLmNvbTovYCxcbiAgICAgIG1vdW50UG9pbnQ6IFwiL21udC9tbFwiLFxuICAgICAgaWRlbnRpZmllcjogXCJtbFwiLFxuICAgICAgbW91bnRPcHRpb25zOiBcIm5mc3ZlcnM9NC4xLHJzaXplPTEwNDg1NzYsd3NpemU9MTA0ODU3NixoYXJkLHRpbWVvPTYwMCxyZXRyYW5zPTJcIlxuICAgIH1dO1xuICAgIGNmblByb2plY3QubG9nc0NvbmZpZyA9IHtcbiAgICAgIGNsb3VkV2F0Y2hMb2dzOiB7XG4gICAgICAgIHN0YXR1czogXCJFTkFCTEVEXCJcbiAgICAgIH1cbiAgICB9O1xuICAgIGNmblByb2plY3QuYWRkUHJvcGVydHlPdmVycmlkZShcbiAgICAgICdFbnZpcm9ubWVudC5JbWFnZVB1bGxDcmVkZW50aWFsc1R5cGUnLFxuICAgICAgJ0NPREVCVUlMRCdcbiAgICApO1xuXG4gICAgLy8gVHJpZ2dlcnMgdGhlIENvZGVCdWlsZCBwcm9qZWN0IHRvIGluc3RhbGwgdGhlIHB5dGhvbiBwYWNrYWdlcyBhbmQgbW9kZWwgdG8gdGhlIEVGUyBmaWxlIHN5c3RlbVxuICAgIGNvbnN0IHRyaWdnZXJCdWlsZFByb2plY3QgPSBuZXcgY3IuQXdzQ3VzdG9tUmVzb3VyY2UodGhpcywgJ1RyaWdnZXJDb2RlQnVpbGQnLCB7XG4gICAgICBvbkNyZWF0ZToge1xuICAgICAgICBzZXJ2aWNlOiAnQ29kZUJ1aWxkJyxcbiAgICAgICAgYWN0aW9uOiAnc3RhcnRCdWlsZCcsXG4gICAgICAgIHBhcmFtZXRlcnM6IHtcbiAgICAgICAgICBwcm9qZWN0TmFtZTogY29kZUJ1aWxkUHJvamVjdC5wcm9qZWN0TmFtZVxuICAgICAgICB9LFxuICAgICAgICBwaHlzaWNhbFJlc291cmNlSWQ6IGNyLlBoeXNpY2FsUmVzb3VyY2VJZC5mcm9tUmVzcG9uc2UoJ2J1aWxkLmlkJyksXG4gICAgICB9LFxuICAgICAgb25VcGRhdGU6IHtcbiAgICAgICAgc2VydmljZTogJ0NvZGVCdWlsZCcsXG4gICAgICAgIGFjdGlvbjogJ3N0YXJ0QnVpbGQnLFxuICAgICAgICBwYXJhbWV0ZXJzOiB7XG4gICAgICAgICAgcHJvamVjdE5hbWU6IGNvZGVCdWlsZFByb2plY3QucHJvamVjdE5hbWVcbiAgICAgICAgfSxcbiAgICAgICAgcGh5c2ljYWxSZXNvdXJjZUlkOiBjci5QaHlzaWNhbFJlc291cmNlSWQuZnJvbVJlc3BvbnNlKCdidWlsZC5pZCcpLFxuICAgICAgfSxcbiAgICAgIHBvbGljeTogY3IuQXdzQ3VzdG9tUmVzb3VyY2VQb2xpY3kuZnJvbVNka0NhbGxzKHsgcmVzb3VyY2VzOiBjci5Bd3NDdXN0b21SZXNvdXJjZVBvbGljeS5BTllfUkVTT1VSQ0UgfSlcbiAgICB9KTtcblxuICAgIC8vIENyZWF0ZSBkZXBlbmRlbmN5IGJldHdlZW4gRUZTIGFuZCBDb2RlYnVpbGRcbiAgICBjb2RlQnVpbGRQcm9qZWN0Lm5vZGUuYWRkRGVwZW5kZW5jeSh0aGlzLmFjY2Vzc1BvaW50KTtcblxuICAgIHRoaXMubGFtYmRhRnVuY3Rpb24gPSBuZXcgbGFtYmRhLkZ1bmN0aW9uKHRoaXMsIFwiTGFtYmRhRnVuY3Rpb25cIiwge1xuICAgICAgaGFuZGxlcjogXCJhcHAubGFtYmRhX2hhbmRsZXJcIixcbiAgICAgIHJ1bnRpbWU6IGxhbWJkYS5SdW50aW1lLlBZVEhPTl8zXzcsXG4gICAgICBjb2RlOiBsYW1iZGEuQ29kZS5mcm9tQXNzZXQocGF0aC5qb2luKF9fZGlybmFtZSwgJy4vc3JjLycpLCB7IC8vIFRPRE86IGRvIHdlIG5lZWQgYWxsIGZpbGVzIGluIHNyYyBmb2xkZXI/XG4gICAgICAgIGJ1bmRsaW5nOiB7XG4gICAgICAgICAgaW1hZ2U6IGxhbWJkYS5SdW50aW1lLlBZVEhPTl8zXzcuYnVuZGxpbmdEb2NrZXJJbWFnZSxcbiAgICAgICAgICBjb21tYW5kOiBbXG4gICAgICAgICAgICAnYmFzaCcsICctYycsIFtcbiAgICAgICAgICAgICAgYGNwIC1yIC9hc3NldC1pbnB1dC8qIC9hc3NldC1vdXRwdXQvYCxcbiAgICAgICAgICAgICAgYGNkIC9hc3NldC1vdXRwdXQvYCxcbiAgICAgICAgICAgICAgYGNobW9kIGEreCAuL3NldHVwLnNoYCxcbiAgICAgICAgICAgICAgYC4vc2V0dXAuc2hgLCBgbHMgLWxhaGBcbiAgICAgICAgICAgIF0uam9pbignICYmICcpXG4gICAgICAgICAgXSxcbiAgICAgICAgICB1c2VyOiAncm9vdCdcbiAgICAgICAgfVxuICAgICAgfSksXG4gICAgICBtZW1vcnlTaXplOiAyMDQ4LFxuICAgICAgdGltZW91dDogY2RrLkR1cmF0aW9uLnNlY29uZHMoMzApLFxuICAgICAgZW52aXJvbm1lbnQ6IHtcbiAgICAgICAgREFUQV9CVUNLRVQ6IHRoaXMuZGF0YUJ1Y2tldC5idWNrZXROYW1lLFxuICAgICAgICBNT0RFTF9CVUNLRVQ6IHRoaXMubW9kZWxCdWNrZXQuYnVja2V0TmFtZSxcbiAgICAgICAgUFJPRF9CVUNLRVQ6IHByb2RCdWNrZXQuYnVja2V0TmFtZSxcbiAgICAgICAgVEFTS19UQUJMRTogdGhpcy50YXNrVGFibGUudGFibGVOYW1lLFxuICAgICAgICBCQVRDSF9PUFNfUk9MRV9BUk46IGJhdGNoT3BzUm9sZS5yb2xlQXJuLFxuICAgICAgICBTQUdFTUFLRVJfRVhFQ19ST0xFX0FSTjogc2FnZW1ha2VyRXhlY1JvbGUucm9sZUFybixcbiAgICAgICAgUFJJVkFURV9TVUJORVRTOiBwcm9wcy5wcml2YXRlU3VibmV0cyxcbiAgICAgICAgVlBDX0lEOiBwcm9wcy52cGMudnBjSWQsXG4gICAgICB9LFxuICAgICAgaW5pdGlhbFBvbGljeTogW1xuICAgICAgICBzM1JlYWRQb2xpY3ksIHMzUG9saWN5LCBzM2NvbnRyb2xQb2xpY3ksIGVjc1BvbGljeSwgc2FnZW1ha2VyUG9saWN5LCBkeW5hbW9kYlBvbGljeSwgYXNnUG9saWN5XG4gICAgICBdLFxuICAgICAgc2VjdXJpdHlHcm91cDogbGFtYmRhU2VjdXJpdHlHcm91cCxcbiAgICAgIHZwYzogcHJvcHMudnBjLFxuICAgICAgZmlsZXN5c3RlbTogbGFtYmRhLkZpbGVTeXN0ZW0uZnJvbUVmc0FjY2Vzc1BvaW50KHRoaXMuYWNjZXNzUG9pbnQsIFwiL21udC9tbFwiKSxcbiAgICB9KVxuXG4gICAgdGhpcy5kYXRhQnVja2V0LmdyYW50UmVhZFdyaXRlKHRoaXMubGFtYmRhRnVuY3Rpb24pO1xuICAgIHRoaXMubW9kZWxCdWNrZXQuZ3JhbnRSZWFkV3JpdGUodGhpcy5sYW1iZGFGdW5jdGlvbik7XG4gICAgcHJvZEJ1Y2tldC5ncmFudFJlYWRXcml0ZSh0aGlzLmxhbWJkYUZ1bmN0aW9uKTtcblxuICAgIGlmIChwcm9wcy5hdXRoVHlwZSA9PT0gQXV0aFR5cGUuQ09HTklUTykge1xuICAgICAgLy8gQ3JlYXRlIENvZ25pdG8gVXNlciBQb29sXG4gICAgICB0aGlzLnVzZXJQb29sID0gbmV3IGNvZ25pdG8uVXNlclBvb2wodGhpcywgJ1VzZXJQb29sJywge1xuICAgICAgICBzZWxmU2lnblVwRW5hYmxlZDogZmFsc2UsXG4gICAgICAgIHNpZ25JbkNhc2VTZW5zaXRpdmU6IGZhbHNlLFxuICAgICAgICBzaWduSW5BbGlhc2VzOiB7XG4gICAgICAgICAgZW1haWw6IHRydWUsXG4gICAgICAgICAgdXNlcm5hbWU6IGZhbHNlLFxuICAgICAgICAgIHBob25lOiB0cnVlXG4gICAgICAgIH1cbiAgICAgIH0pXG5cbiAgICAgIC8vIENyZWF0ZSBVc2VyIFBvb2wgQ2xpZW50XG4gICAgICB0aGlzLnVzZXJQb29sQXBpQ2xpZW50ID0gbmV3IGNvZ25pdG8uVXNlclBvb2xDbGllbnQodGhpcywgJ1VzZXJQb29sQXBpQ2xpZW50Jywge1xuICAgICAgICB1c2VyUG9vbDogdGhpcy51c2VyUG9vbCxcbiAgICAgICAgdXNlclBvb2xDbGllbnROYW1lOiAnUmVwbGljYXRpb25IdWJQb3J0YWwnLFxuICAgICAgICBwcmV2ZW50VXNlckV4aXN0ZW5jZUVycm9yczogdHJ1ZVxuICAgICAgfSlcbiAgICB9XG5cbiAgICBjb25zdCBsYW1iZGFGbiA9IG5ldyBhcGlnYXRld2F5LkxhbWJkYUludGVncmF0aW9uKHRoaXMubGFtYmRhRnVuY3Rpb24sIHt9KTtcbiAgICBjb25zdCBjdXN0b21PcHRpb25zID0ge1xuICAgICAgYXV0aG9yaXphdGlvblR5cGU6IGFwaWdhdGV3YXkuQXV0aG9yaXphdGlvblR5cGUuQ1VTVE9NLFxuICAgICAgYXV0aG9yaXplcjogbmV3IGFwaWdhdGV3YXkuVG9rZW5BdXRob3JpemVyKHRoaXMsIFwiVG9rZW5BdXRob3JpemVyXCIsIHtcbiAgICAgICAgaGFuZGxlcjogdGhpcy5sYW1iZGFGdW5jdGlvbixcbiAgICAgICAgaWRlbnRpdHlTb3VyY2U6IFwibWV0aG9kLnJlcXVlc3QuaGVhZGVyLmF1dGhvcml6YXRpb25cIixcbiAgICAgICAgcmVzdWx0c0NhY2hlVHRsOiBjZGsuRHVyYXRpb24ubWludXRlcyg1KVxuICAgICAgfSlcbiAgICB9XG4gICAgY29uc3QgbWV0aG9kT3B0aW9ucyA9IGN1c3RvbU9wdGlvbnM7XG5cbiAgICBjb25zdCB0YXNrQXBpID0gdGhpcy5yZXN0QXBpLnJvb3QuYWRkUmVzb3VyY2UoJ3Rhc2tzJyk7XG4gICAgdGFza0FwaS5hZGRNZXRob2QoJ0dFVCcsIGxhbWJkYUZuLCBtZXRob2RPcHRpb25zKTtcbiAgICB0YXNrQXBpLmFkZE1ldGhvZCgnUE9TVCcsIGxhbWJkYUZuLCBtZXRob2RPcHRpb25zKTtcbiAgICBjb25zdCB0YXNrSWRBcGkgPSB0YXNrQXBpLmFkZFJlc291cmNlKCd7dGFza19pZH0nKTtcbiAgICB0YXNrSWRBcGkuYWRkTWV0aG9kKCdERUxFVEUnLCBsYW1iZGFGbiwgbWV0aG9kT3B0aW9ucyk7XG4gICAgdGFza0lkQXBpLmFkZE1ldGhvZCgnR0VUJywgbGFtYmRhRm4sIG1ldGhvZE9wdGlvbnMpO1xuICAgIGNvbnN0IHRhc2tTdG9wQXBpID0gdGFza0lkQXBpLmFkZFJlc291cmNlKCdzdG9wJyk7XG4gICAgdGFza1N0b3BBcGkuYWRkTWV0aG9kKCdQT1NUJywgbGFtYmRhRm4sIG1ldGhvZE9wdGlvbnMpO1xuICAgIGNvbnN0IHRhc2tEYXRhQXBpID0gdGFza0lkQXBpLmFkZFJlc291cmNlKCdkYXRhJyk7XG4gICAgdGFza0RhdGFBcGkuYWRkTWV0aG9kKCdHRVQnLCBsYW1iZGFGbiwgbWV0aG9kT3B0aW9ucyk7XG4gICAgY29uc3QgdGFza1N0YXR1c0FwaSA9IHRhc2tJZEFwaS5hZGRSZXNvdXJjZSgnc3RhdHVzJyk7XG4gICAgdGFza1N0YXR1c0FwaS5hZGRNZXRob2QoJ0dFVCcsIGxhbWJkYUZuLCBtZXRob2RPcHRpb25zKTtcbiAgICBjb25zdCB0YXNrRGF0YUNsYXNzSWRBcGkgPSB0YXNrRGF0YUFwaS5hZGRSZXNvdXJjZSgne2NsYXNzX2lkfScpO1xuICAgIHRhc2tEYXRhQ2xhc3NJZEFwaS5hZGRNZXRob2QoJ0dFVCcsIGxhbWJkYUZuLCBtZXRob2RPcHRpb25zKTtcbiAgICB0YXNrRGF0YUNsYXNzSWRBcGkuYWRkTWV0aG9kKCdQT1NUJywgbGFtYmRhRm4sIG1ldGhvZE9wdGlvbnMpO1xuICAgIGNvbnN0IHRhc2tTM0RhdGFBcGkgPSB0YXNrSWRBcGkuYWRkUmVzb3VyY2UoJ3MzZGF0YScpO1xuICAgIHRhc2tTM0RhdGFBcGkuYWRkTWV0aG9kKCdHRVQnLCBsYW1iZGFGbiwgbWV0aG9kT3B0aW9ucyk7XG4gICAgdGFza1MzRGF0YUFwaS5hZGRNZXRob2QoJ1BPU1QnLCBsYW1iZGFGbiwgbWV0aG9kT3B0aW9ucyk7XG4gICAgY29uc3QgdGFza1RyYWluQXBpID0gdGFza0lkQXBpLmFkZFJlc291cmNlKCd0cmFpbicpO1xuICAgIHRhc2tUcmFpbkFwaS5hZGRNZXRob2QoJ1BPU1QnLCBsYW1iZGFGbiwgbWV0aG9kT3B0aW9ucyk7XG4gICAgY29uc3QgdGFza1ByZWRpY3RBcGkgPSB0YXNrSWRBcGkuYWRkUmVzb3VyY2UoJ3ByZWRpY3QnKTtcbiAgICB0YXNrUHJlZGljdEFwaS5hZGRNZXRob2QoJ1BPU1QnLCBsYW1iZGFGbiwgbWV0aG9kT3B0aW9ucyk7XG4gICAgY29uc3QgdGFza1ByZWRpY3RWMkFwaSA9IHRhc2tJZEFwaS5hZGRSZXNvdXJjZSgncHJlZGljdF92MicpO1xuICAgIHRhc2tQcmVkaWN0VjJBcGkuYWRkTWV0aG9kKCdQT1NUJywgbGFtYmRhRm4sIG1ldGhvZE9wdGlvbnMpO1xuICAgIGNvbnN0IHRhc2tEZXBsb3lBcGkgPSB0YXNrSWRBcGkuYWRkUmVzb3VyY2UoJ2RlcGxveScpO1xuICAgIHRhc2tEZXBsb3lBcGkuYWRkTWV0aG9kKCdQT1NUJywgbGFtYmRhRm4sIG1ldGhvZE9wdGlvbnMpO1xuXG4gICAgYWRkQ29yc09wdGlvbnModGFza0FwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza0lkQXBpKTtcbiAgICBhZGRDb3JzT3B0aW9ucyh0YXNrU3RvcEFwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza1N0YXR1c0FwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza0RhdGFBcGkpO1xuICAgIGFkZENvcnNPcHRpb25zKHRhc2tEYXRhQ2xhc3NJZEFwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza1MzRGF0YUFwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza1RyYWluQXBpKTtcbiAgICBhZGRDb3JzT3B0aW9ucyh0YXNrUHJlZGljdEFwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza1ByZWRpY3RWMkFwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza0RlcGxveUFwaSk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGFkZENvcnNPcHRpb25zKGFwaVJlc291cmNlOiBhcGlnYXRld2F5LklSZXNvdXJjZSkge1xuICBhcGlSZXNvdXJjZS5hZGRNZXRob2QoJ09QVElPTlMnLCBuZXcgYXBpZ2F0ZXdheS5Nb2NrSW50ZWdyYXRpb24oe1xuICAgIGludGVncmF0aW9uUmVzcG9uc2VzOiBbe1xuICAgICAgc3RhdHVzQ29kZTogJzIwMCcsXG4gICAgICByZXNwb25zZVBhcmFtZXRlcnM6IHtcbiAgICAgICAgJ21ldGhvZC5yZXNwb25zZS5oZWFkZXIuQWNjZXNzLUNvbnRyb2wtQWxsb3ctSGVhZGVycyc6IFwiJ0NvbnRlbnQtVHlwZSxYLUFtei1EYXRlLEF1dGhvcml6YXRpb24sWC1BcGktS2V5LFgtQW16LVNlY3VyaXR5LVRva2VuLFgtQW16LVVzZXItQWdlbnQnXCIsXG4gICAgICAgICdtZXRob2QucmVzcG9uc2UuaGVhZGVyLkFjY2Vzcy1Db250cm9sLUFsbG93LU9yaWdpbic6IFwiJyonXCIsXG4gICAgICAgICdtZXRob2QucmVzcG9uc2UuaGVhZGVyLkFjY2Vzcy1Db250cm9sLUFsbG93LUNyZWRlbnRpYWxzJzogXCInZmFsc2UnXCIsXG4gICAgICAgICdtZXRob2QucmVzcG9uc2UuaGVhZGVyLkFjY2Vzcy1Db250cm9sLUFsbG93LU1ldGhvZHMnOiBcIidPUFRJT05TLEdFVCxQVVQsUE9TVCxERUxFVEUnXCIsXG4gICAgICB9LFxuICAgIH1dLFxuICAgIHBhc3N0aHJvdWdoQmVoYXZpb3I6IGFwaWdhdGV3YXkuUGFzc3Rocm91Z2hCZWhhdmlvci5ORVZFUixcbiAgICByZXF1ZXN0VGVtcGxhdGVzOiB7XG4gICAgICBcImFwcGxpY2F0aW9uL2pzb25cIjogXCJ7XFxcInN0YXR1c0NvZGVcXFwiOiAyMDB9XCJcbiAgICB9LFxuICB9KSwge1xuICAgIG1ldGhvZFJlc3BvbnNlczogW3tcbiAgICAgIHN0YXR1c0NvZGU6ICcyMDAnLFxuICAgICAgcmVzcG9uc2VQYXJhbWV0ZXJzOiB7XG4gICAgICAgICdtZXRob2QucmVzcG9uc2UuaGVhZGVyLkFjY2Vzcy1Db250cm9sLUFsbG93LUhlYWRlcnMnOiB0cnVlLFxuICAgICAgICAnbWV0aG9kLnJlc3BvbnNlLmhlYWRlci5BY2Nlc3MtQ29udHJvbC1BbGxvdy1NZXRob2RzJzogdHJ1ZSxcbiAgICAgICAgJ21ldGhvZC5yZXNwb25zZS5oZWFkZXIuQWNjZXNzLUNvbnRyb2wtQWxsb3ctQ3JlZGVudGlhbHMnOiB0cnVlLFxuICAgICAgICAnbWV0aG9kLnJlc3BvbnNlLmhlYWRlci5BY2Nlc3MtQ29udHJvbC1BbGxvdy1PcmlnaW4nOiB0cnVlLFxuICAgICAgfSxcbiAgICB9XSxcbiAgfSlcbn1cbiJdfQ==