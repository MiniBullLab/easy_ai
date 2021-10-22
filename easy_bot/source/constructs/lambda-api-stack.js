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
                            'chown -R 1000:1000 /mnt/ml/',
                            'curl -O http://118.31.19.101:8080/software/easyai/easyai-0.3-py2.py3-none-any.whl',
                            '/mnt/ml/code/bin/pip3 install -i https://mirrors.aliyun.com/pypi/simple easyai-0.3-py2.py3-none-any.whl',
                            'rm -rf easyai-0.3-py2.py3-none-any.whl'
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibGFtYmRhLWFwaS1zdGFjay5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbImxhbWJkYS1hcGktc3RhY2sudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7O0FBQUEsd0NBQXVDO0FBQ3ZDLDhDQUE4QztBQUU5QyxxQ0FBcUM7QUFDckMsc0RBQXNEO0FBQ3RELGtEQUFrRDtBQUNsRCw2QkFBNkI7QUFDN0Isc0NBQXNDO0FBQ3RDLGdEQUFnRDtBQUNoRCx3Q0FBd0M7QUFDeEMsd0NBQXdDO0FBQ3hDLG9EQUFtRDtBQUNuRCxnREFBK0M7QUFTL0MsTUFBYSxjQUFlLFNBQVEsR0FBRyxDQUFDLFNBQVM7SUFnQi9DLFlBQVksTUFBcUIsRUFBRSxFQUFVLEVBQUUsS0FBMEI7UUFDdkUsS0FBSyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsQ0FBQTtRQUVqQixJQUFJLENBQUMsU0FBUyxHQUFHLElBQUksUUFBUSxDQUFDLEtBQUssQ0FBQyxJQUFJLEVBQUUsV0FBVyxFQUFFO1lBQ3JELFlBQVksRUFBRSxFQUFFLElBQUksRUFBRSxPQUFPLEVBQUUsSUFBSSxFQUFFLFFBQVEsQ0FBQyxhQUFhLENBQUMsTUFBTSxFQUFFO1lBQ3BFLFdBQVcsRUFBRSxRQUFRLENBQUMsV0FBVyxDQUFDLGVBQWU7WUFDakQsYUFBYSxFQUFFLEdBQUcsQ0FBQyxhQUFhLENBQUMsT0FBTztTQUN6QyxDQUFDLENBQUM7UUFFSCxrRUFBa0U7UUFDbEUsSUFBSSxDQUFDLFVBQVUsR0FBRyxJQUFJLEVBQUUsQ0FBQyxNQUFNLENBQUMsSUFBSSxFQUFFLFlBQVksRUFBRTtZQUNsRCxhQUFhLEVBQUUsR0FBRyxDQUFDLGFBQWEsQ0FBQyxPQUFPO1lBQ3hDLGlCQUFpQixFQUFFLElBQUk7U0FDeEIsQ0FBQyxDQUFBO1FBQ0YsSUFBSSxDQUFDLFdBQVcsR0FBRyxJQUFJLEVBQUUsQ0FBQyxNQUFNLENBQUMsSUFBSSxFQUFFLGFBQWEsRUFBRTtZQUNwRCxhQUFhLEVBQUUsR0FBRyxDQUFDLGFB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RSxDQUFDO2dCQUNyQixVQUFVLEVBQUUsS0FBSztnQkFDakIsa0JBQWtCLEVBQUU7b0JBQ2xCLHFEQUFxRCxFQUFFLHlGQUF5RjtvQkFDaEosb0RBQW9ELEVBQUUsS0FBSztvQkFDM0QseURBQXlELEVBQUUsU0FBUztvQkFDcEUscURBQXFELEVBQUUsK0JBQStCO2lCQUN2RjthQUNGLENBQUM7UUFDRixtQkFBbUIsRUFBRSxVQUFVLENBQUMsbUJBQW1CLENBQUMsS0FBSztRQUN6RCxnQkFBZ0IsRUFBRTtZQUNoQixrQkFBa0IsRUFBRSx1QkFBdUI7U0FDNUM7S0FDRixDQUFDLEVBQUU7UUFDRixlQUFlLEVBQUUsQ0FBQztnQkFDaEIsVUFBVSxFQUFFLEtBQUs7Z0JBQ2pCLGtCQUFrQixFQUFFO29CQUNsQixxREFBcUQsRUFBRSxJQUFJO29CQUMzRCxxREFBcUQsRUFBRSxJQUFJO29CQUMzRCx5REFBeUQsRUFBRSxJQUFJO29CQUMvRCxvREFBb0QsRUFBRSxJQUFJO2lCQUMzRDthQUNGLENBQUM7S0FDSCxDQUFDLENBQUE7QUFDSixDQUFDO0FBMUJELHdDQTBCQyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIGlhbSBmcm9tIFwiQGF3cy1jZGsvYXdzLWlhbVwiXG5pbXBvcnQgKiBhcyBsYW1iZGEgZnJvbSBcIkBhd3MtY2RrL2F3cy1sYW1iZGFcIjtcbmltcG9ydCB7VnBjfSBmcm9tIFwiQGF3cy1jZGsvYXdzLWVjMlwiXG5pbXBvcnQgKiBhcyBjZGsgZnJvbSBcIkBhd3MtY2RrL2NvcmVcIjtcbmltcG9ydCAqIGFzIGFwaWdhdGV3YXkgZnJvbSAnQGF3cy1jZGsvYXdzLWFwaWdhdGV3YXknO1xuaW1wb3J0ICogYXMgZHluYW1vZGIgZnJvbSAnQGF3cy1jZGsvYXdzLWR5bmFtb2RiJztcbmltcG9ydCAqIGFzIHBhdGggZnJvbSAncGF0aCc7XG5pbXBvcnQgKiBhcyBzMyBmcm9tICdAYXdzLWNkay9hd3MtczMnO1xuaW1wb3J0ICogYXMgY29nbml0byBmcm9tICdAYXdzLWNkay9hd3MtY29nbml0byc7XG5pbXBvcnQgKiBhcyBlZnMgZnJvbSAnQGF3cy1jZGsvYXdzLWVmcyc7XG5pbXBvcnQgKiBhcyBlYzIgZnJvbSAnQGF3cy1jZGsvYXdzLWVjMic7XG5pbXBvcnQgKiBhcyBjb2RlYnVpbGQgZnJvbSBcIkBhd3MtY2RrL2F3cy1jb2RlYnVpbGRcIlxuaW1wb3J0ICogYXMgY3IgZnJvbSBcIkBhd3MtY2RrL2N1c3RvbS1yZXNvdXJjZXNcIlxuaW1wb3J0IHsgQXV0aFR5cGUgfSBmcm9tICcuL2luZGV4JztcblxuZXhwb3J0IGludGVyZmFjZSBMYW1iZGFBcGlTdGFja1Byb3BzIHtcbiAgcmVhZG9ubHkgYXV0aFR5cGU6IEF1dGhUeXBlO1xuICByZWFkb25seSB2cGM6IGVjMi5JVnBjO1xuICByZWFkb25seSBwcml2YXRlU3VibmV0czogc3RyaW5nO1xufVxuXG5leHBvcnQgY2xhc3MgTGFtYmRhQXBpU3RhY2sgZXh0ZW5kcyBjZGsuQ29uc3RydWN0IHtcbiAgcmVhZG9ubHkgcmVzdEFwaTogYXBpZ2F0ZXdheS5SZXN0QXBpXG4gIHJlYWRvbmx5IGxhbWJkYUZ1bmN0aW9uOiBsYW1iZGEuRnVuY3Rpb25cbiAgcmVhZG9ubHkgZGF0YUJ1Y2tldDogczMuQnVja2V0XG4gIHJlYWRvbmx5IG1vZGVsQnVja2V0OiBzMy5CdWNrZXRcbiAgLy8gcmVhZG9ubHkgdnBjOiBWcGNcbiAgcmVhZG9ubHkgdGFza1RhYmxlOiBkeW5hbW9kYi5UYWJsZVxuXG4gIHJlYWRvbmx5IHVzZXJQb29sPzogY29nbml0by5Vc2VyUG9vbFxuICByZWFkb25seSB1c2VyUG9vbEFwaUNsaWVudD86IGNvZ25pdG8uVXNlclBvb2xDbGllbnRcbiAgcmVhZG9ubHkgdXNlclBvb2xEb21haW4/OiBjb2duaXRvLlVzZXJQb29sRG9tYWluXG5cbiAgcmVhZG9ubHkgZmlsZVN5c3RlbSA6IGVmcy5GaWxlU3lzdGVtXG4gIHJlYWRvbmx5IGFjY2Vzc1BvaW50IDogZWZzLkFjY2Vzc1BvaW50XG4gIHJlYWRvbmx5IGVjMlNlY3VyaXR5R3JvdXAgOiBlYzIuU2VjdXJpdHlHcm91cFxuXG4gIGNvbnN0cnVjdG9yKHBhcmVudDogY2RrLkNvbnN0cnVjdCwgaWQ6IHN0cmluZywgcHJvcHM6IExhbWJkYUFwaVN0YWNrUHJvcHMpIHtcbiAgICBzdXBlcihwYXJlbnQsIGlkKVxuXG4gICAgdGhpcy50YXNrVGFibGUgPSBuZXcgZHluYW1vZGIuVGFibGUodGhpcywgJ1Rhc2tUYWJsZScsIHtcbiAgICAgIHBhcnRpdGlvbktleTogeyBuYW1lOiAnSm9iSWQnLCB0eXBlOiBkeW5hbW9kYi5BdHRyaWJ1dGVUeXBlLlNUUklORyB9LFxuICAgICAgYmlsbGluZ01vZGU6IGR5bmFtb2RiLkJpbGxpbmdNb2RlLlBBWV9QRVJfUkVRVUVTVCxcbiAgICAgIHJlbW92YWxQb2xpY3k6IGNkay5SZW1vdmFsUG9saWN5LkRFU1RST1lcbiAgICB9KTtcbiAgICBcbiAgICAvLyB0aGlzLnZwYyA9IG5ldyBWcGModGhpcywgJ1ZwYycsIHsgbWF4QXpzOiAyLCBuYXRHYXRld2F5czogMSB9KTtcbiAgICB0aGlzLmRhdGFCdWNrZXQgPSBuZXcgczMuQnVja2V0KHRoaXMsIFwiRGF0YUJ1Y2tldFwiLCB7XG4gICAgICByZW1vdmFsUG9saWN5OiBjZGsuUmVtb3ZhbFBvbGljeS5ERVNUUk9ZLFxuICAgICAgYXV0b0RlbGV0ZU9iamVjdHM6IHRydWVcbiAgICB9KVxuICAgIHRoaXMubW9kZWxCdWNrZXQgPSBuZXcgczMuQnVja2V0KHRoaXMsIFwiTW9kZWxCdWNrZXRcIiwge1xuICAgICAgcmVtb3ZhbFBvbGljeTogY2RrLlJlbW92YWxQb2xpY3kuREVTVFJPWSxcbiAgICAgIGF1dG9EZWxldGVPYmplY3RzOiB0cnVlXG4gICAgfSlcbiAgICBjb25zdCBwcm9kQnVja2V0ID0gbmV3IHMzLkJ1Y2tldCh0aGlzLCBcIlByb2RCdWNrZXRcIiwge1xuICAgICAgcmVtb3ZhbFBvbGljeTogY2RrLlJlbW92YWxQb2xpY3kuREVTVFJPWSxcbiAgICAgIGF1dG9EZWxldGVPYmplY3RzOiBmYWxzZVxuICAgIH0pXG5cbiAgICB0aGlzLnJlc3RBcGkgPSBuZXcgYXBpZ2F0ZXdheS5SZXN0QXBpKHRoaXMsIFwiUmVzdEFwaVwiLCB7XG4gICAgICByZXN0QXBpTmFtZTogY2RrLkF3cy5TVEFDS19OQU1FLFxuICAgICAgZGVwbG95T3B0aW9uczoge1xuICAgICAgICBzdGFnZU5hbWU6IFwiUHJvZFwiLFxuICAgICAgICBtZXRyaWNzRW5hYmxlZDogdHJ1ZSxcbiAgICAgICAgbG9nZ2luZ0xldmVsOiBhcGlnYXRld2F5Lk1ldGhvZExvZ2dpbmdMZXZlbC5JTkZPLFxuICAgICAgICBkYXRhVHJhY2VFbmFibGVkOiB0cnVlLFxuICAgICAgfSxcbiAgICAgIGVuZHBvaW50Q29uZmlndXJhdGlvbjoge1xuICAgICAgICB0eXBlczogWyBhcGlnYXRld2F5LkVuZHBvaW50VHlwZS5SRUdJT05BTCBdXG4gICAgICB9XG4gICAgfSlcblxuICAgIC8vIFNlY3VyaXR5IEdyb3VwIGRlZmluaXRpb25zLlxuICAgIHRoaXMuZWMyU2VjdXJpdHlHcm91cCA9IG5ldyBlYzIuU2VjdXJpdHlHcm91cCh0aGlzLCAnTGFtYmRhRUZTTUxFQzJTRycsIHtcbiAgICAgIHZwYzogcHJvcHMudnBjLCBhbGxvd0FsbE91dGJvdW5kOiB0cnVlLFxuICAgIH0pO1xuICAgIGNvbnN0IGxhbWJkYVNlY3VyaXR5R3JvdXAgPSBuZXcgZWMyLlNlY3VyaXR5R3JvdXAodGhpcywgJ0xhbWJkYUVGU01MTGFtYmRhU0cnLCB7XG4gICAgICB2cGM6IHByb3BzLnZwYywgYWxsb3dBbGxPdXRib3VuZDogdHJ1ZSxcbiAgICB9KTtcbiAgICBjb25zdCBlZnNTZWN1cml0eUdyb3VwID0gbmV3IGVjMi5TZWN1cml0eUdyb3VwKHRoaXMsICdMYW1iZGFFRlNNTEVGU1NHJywge1xuICAgICAgdnBjOiBwcm9wcy52cGMsIGFsbG93QWxsT3V0Ym91bmQ6IHRydWUsXG4gICAgfSk7XG4gICAgdGhpcy5lYzJTZWN1cml0eUdyb3VwLmNvbm5lY3Rpb25zLmFsbG93VG8oZWZzU2VjdXJpdHlHcm91cCwgZWMyLlBvcnQudGNwKDIwNDkpKTtcbiAgICBsYW1iZGFTZWN1cml0eUdyb3VwLmNvbm5lY3Rpb25zLmFsbG93VG8oZWZzU2VjdXJpdHlHcm91cCwgZWMyLlBvcnQudGNwKDIwNDkpKTtcbiAgICBcbiAgICBjb25zdCBzM1JlYWRQb2xpY3kgPSBuZXcgaWFtLlBvbGljeVN0YXRlbWVudCgpXG4gICAgczNSZWFkUG9saWN5LmFkZEFjdGlvbnMoXCJzMzpHZXRPYmplY3QqXCIpXG4gICAgczNSZWFkUG9saWN5LmFkZEFjdGlvbnMoXCJzMzpMaXN0QnVja2V0XCIpXG4gICAgczNSZWFkUG9saWN5LmFkZFJlc291cmNlcyhcIipcIilcbiAgICBjb25zdCBzM1BvbGljeSA9IG5ldyBpYW0uUG9saWN5U3RhdGVtZW50KClcbiAgICBzM1BvbGljeS5hZGRBY3Rpb25zKFwiczM6KlwiKVxuICAgIHMzUG9saWN5LmFkZFJlc291cmNlcyh0aGlzLmRhdGFCdWNrZXQuYnVja2V0QXJuICsgXCIvKlwiKVxuICAgIHMzUG9saWN5LmFkZFJlc291cmNlcyh0aGlzLm1vZGVsQnVja2V0LmJ1Y2tldEFybiArIFwiLypcIilcbiAgICBzM1BvbGljeS5hZGRSZXNvdXJjZXMocHJvZEJ1Y2tldC5idWNrZXRBcm4gKyBcIi8qXCIpXG4gICAgY29uc3QgczNjb250cm9sUG9saWN5ID0gbmV3IGlhbS5Qb2xpY3lTdGF0ZW1lbnQoKVxuICAgIHMzY29udHJvbFBvbGljeS5hZGRBY3Rpb25zKFwiczM6KlwiKVxuICAgIHMzY29udHJvbFBvbGljeS5hZGRBY3Rpb25zKFwiaWFtOipcIilcbiAgICBzM2NvbnRyb2xQb2xpY3kuYWRkUmVzb3VyY2VzKFwiKlwiKVxuICAgIGNvbnN0IGVjc1BvbGljeSA9IG5ldyBpYW0uUG9saWN5U3RhdGVtZW50KClcbiAgICBlY3NQb2xpY3kuYWRkQWN0aW9ucyhcImVjMjoqXCIpXG4gICAgZWNzUG9saWN5LmFkZEFjdGlvbnMoXCJlY3M6KlwiKVxuICAgIGVjc1BvbGljeS5hZGRBY3Rpb25zKFwiaWFtOipcIilcbiAgICBlY3NQb2xpY3kuYWRkUmVzb3VyY2VzKFwiKlwiKVxuICAgIGNvbnN0IHNhZ2VtYWtlclBvbGljeSA9IG5ldyBpYW0uUG9saWN5U3RhdGVtZW50KClcbiAgICBzYWdlbWFrZXJQb2xpY3kuYWRkQWN0aW9ucyhcInNhZ2VtYWtlcjoqXCIpXG4gICAgc2FnZW1ha2VyUG9saWN5LmFkZFJlc291cmNlcyhcIipcIilcbiAgICBjb25zdCBkeW5hbW9kYlBvbGljeSA9IG5ldyBpYW0uUG9saWN5U3RhdGVtZW50KClcbiAgICBkeW5hbW9kYlBvbGljeS5hZGRBY3Rpb25zKFwiZHluYW1vZGI6KlwiKVxuICAgIGR5bmFtb2RiUG9saWN5LmFkZEFjdGlvbnMoXCJpYW06KlwiKVxuICAgIGR5bmFtb2RiUG9saWN5LmFkZFJlc291cmNlcyh0aGlzLnRhc2tUYWJsZS50YWJsZUFybilcbiAgICBjb25zdCBhc2dQb2xpY3kgPSBuZXcgaWFtLlBvbGljeVN0YXRlbWVudCgpXG4gICAgYXNnUG9saWN5LmFkZEFjdGlvbnMoXCJhdXRvc2NhbGluZzoqXCIpXG4gICAgYXNnUG9saWN5LmFkZEFjdGlvbnMoXCJlbGFzdGljbG9hZGJhbGFuY2luZzoqXCIpXG4gICAgYXNnUG9saWN5LmFkZEFjdGlvbnMoXCJjbG91ZHdhdGNoOipcIilcbiAgICBhc2dQb2xpY3kuYWRkUmVzb3VyY2VzKFwiKlwiKVxuXG4gICAgY29uc3QgYmF0Y2hPcHNSb2xlID0gbmV3IGlhbS5Sb2xlKHRoaXMsIFwiQWRtaW5Sb2xlRm9yUzNCYXRjaE9wZXJhdGlvbnNcIiwge1xuICAgICAgYXNzdW1lZEJ5OiBuZXcgaWFtLlNlcnZpY2VQcmluY2lwYWwoJ2JhdGNob3BlcmF0aW9ucy5zMy5hbWF6b25hd3MuY29tJyksXG4gICAgICBkZXNjcmlwdGlvbjogXCJBbGxvd3MgUzMgQmF0Y2ggT3BlcmF0aW9ucyB0byBjYWxsIEFXUyBzZXJ2aWNlcyBvbiB5b3VyIGJlaGFsZi5cIixcbiAgICAgIG1hbmFnZWRQb2xpY2llczogW1xuICAgICAgICBpYW0uTWFuYWdlZFBvbGljeS5mcm9tQXdzTWFuYWdlZFBvbGljeU5hbWUoXCJBbWF6b25TM0Z1bGxBY2Nlc3NcIilcbiAgICAgIF1cbiAgICB9KVxuXG4gICAgY29uc3Qgc2FnZW1ha2VyRXhlY1JvbGUgPSBuZXcgaWFtLlJvbGUodGhpcywgXCJBZG1pblJvbGVGb3JTYWdlTWFrZXJFeGVjdXRpb25cIiwge1xuICAgICAgYXNzdW1lZEJ5OiBuZXcgaWFtLlNlcnZpY2VQcmluY2lwYWwoJ3NhZ2VtYWtlci5hbWF6b25hd3MuY29tJyksXG4gICAgICBkZXNjcmlwdGlvbjogXCJBbGxvd3MgU2FnZU1ha2VyIEVuZHBvaW50cyB0byBjYWxsIEFXUyBzZXJ2aWNlcyBvbiB5b3VyIGJlaGFsZi5cIixcbiAgICAgIG1hbmFnZWRQb2xpY2llczogW1xuICAgICAgICBpYW0uTWFuYWdlZFBvbGljeS5mcm9tQXdzTWFuYWdlZFBvbGljeU5hbWUoXCJBbWF6b25TM0Z1bGxBY2Nlc3NcIiksXG4gICAgICAgIGlhbS5NYW5hZ2VkUG9saWN5LmZyb21Bd3NNYW5hZ2VkUG9saWN5TmFtZShcIkFtYXpvbkVDMkNvbnRhaW5lclJlZ2lzdHJ5RnVsbEFjY2Vzc1wiKSxcbiAgICAgICAgaWFtLk1hbmFnZWRQb2xpY3kuZnJvbUF3c01hbmFnZWRQb2xpY3lOYW1lKFwiQW1hem9uU2FnZU1ha2VyRnVsbEFjY2Vzc1wiKSxcbiAgICAgIF1cbiAgICB9KVxuXG4gICAgLy8gRWxhc3RpYyBGaWxlIFN5c3RlbSBmaWxlIHN5c3RlbS5cbiAgICAvLyBGb3IgdGhlIHB1cnBvc2Ugb2YgY29zdCBzYXZpbmcsIHByb3Zpc2lvbmVkIHRocm91Z2hwdXQgaGFzIGJlZW4ga2VwdCBsb3cuXG4gICAgdGhpcy5maWxlU3lzdGVtID0gbmV3IGVmcy5GaWxlU3lzdGVtKHRoaXMsICdGaWxlU3lzdGVtJywge1xuICAgICAgdnBjOiBwcm9wcy52cGMsXG4gICAgICBzZWN1cml0eUdyb3VwOiBlZnNTZWN1cml0eUdyb3VwLFxuICAgICAgdGhyb3VnaHB1dE1vZGU6IGVmcy5UaHJvdWdocHV0TW9kZS5QUk9WSVNJT05FRCxcbiAgICAgIHByb3Zpc2lvbmVkVGhyb3VnaHB1dFBlclNlY29uZDogY2RrLlNpemUubWViaWJ5dGVzKDEwKSxcbiAgICAgIHJlbW92YWxQb2xpY3k6IGNkay5SZW1vdmFsUG9saWN5LkRFU1RST1lcbiAgICB9KTtcbiAgICB0aGlzLmFjY2Vzc1BvaW50ID0gbmV3IGVmcy5BY2Nlc3NQb2ludCh0aGlzLCAnQWNjZXNzUG9pbnQnLCB7XG4gICAgICBmaWxlU3lzdGVtOiB0aGlzLmZpbGVTeXN0ZW0sXG4gICAgICBwYXRoOiAnLycsICAgICAgICAgICAgLy8gcmVtb3ZlIEVGUyBhY2Nlc3MgcG9pbnRcbiAgICAgIHBvc2l4VXNlcjoge1xuICAgICAgICBnaWQ6ICcwJyxcbiAgICAgICAgdWlkOiAnMCdcbiAgICAgIH0sXG4gICAgICBjcmVhdGVBY2w6IHtcbiAgICAgICAgb3duZXJHaWQ6ICcwJyxcbiAgICAgICAgb3duZXJVaWQ6ICcwJyxcbiAgICAgICAgcGVybWlzc2lvbnM6ICc3NzcnXG4gICAgICB9XG4gICAgfSlcblxuICAgIC8vIExldmVyYWdpbmcgb24gQVdTIENvZGVCdWlsZCB0byBpbnN0YWxsIFB5dGhvbiBsaWJyYXJpZXMgdG8gRUZTLlxuICAgIGNvbnN0IGNvZGVCdWlsZFByb2plY3QgPSBuZXcgY29kZWJ1aWxkLlByb2plY3QodGhpcywgJ0xhbWJkYUVGU01MQ29kZUJ1aWxkUHJvamVjdCcsIHtcbiAgICAgIHByb2plY3ROYW1lOiBcIkxhbWJkYUVGU01MQ29kZUJ1aWxkUHJvamVjdFwiLFxuICAgICAgZGVzY3JpcHRpb246IFwiSW5zdGFsbHMgUHl0aG9uIGxpYnJhcmllcyB0byBFRlMuXCIsXG4gICAgICB2cGM6IHByb3BzLnZwYyxcbiAgICAgIGJ1aWxkU3BlYzogY29kZWJ1aWxkLkJ1aWxkU3BlYy5mcm9tT2JqZWN0KHtcbiAgICAgICAgdmVyc2lvbjogJzAuMScsXG4gICAgICAgIHBoYXNlczoge1xuICAgICAgICAgIGJ1aWxkOiB7XG4gICAgICAgICAgICBjb21tYW5kczogW1xuICAgICAgICAgICAgICAnbWtkaXIgLXAgL21udC9tbCcsXG4gICAgICAgICAgICAgICdta2RpciAtcCAvbW50L21sL21vZGVsJyxcbiAgICAgICAgICAgICAgJ3B5dGhvbjMgLW0gdmVudiAvbW50L21sL2NvZGUnLFxuICAgICAgICAgICAgICAnY2hvd24gLVIgMTAwMDoxMDAwIC9tbnQvbWwvJyxcbiAgICAgICAgICAgICAgJ2N1cmwgLU8gaHR0cDovLzExOC4zMS4xOS4xMDE6ODA4MC9zb2Z0d2FyZS9lYXN5YWkvZWFzeWFpLTAuMy1weTIucHkzLW5vbmUtYW55LndobCcsXG4gICAgICAgICAgICAgICcvbW50L21sL2NvZGUvYmluL3BpcDMgaW5zdGFsbCAtaSBodHRwczovL21pcnJvcnMuYWxpeXVuLmNvbS9weXBpL3NpbXBsZSBlYXN5YWktMC4zLXB5Mi5weTMtbm9uZS1hbnkud2hsJyxcbiAgICAgICAgICAgICAgJ3JtIC1yZiBlYXN5YWktMC4zLXB5Mi5weTMtbm9uZS1hbnkud2hsJ1xuICAgICAgICAgICAgXVxuICAgICAgICAgIH1cbiAgICAgICAgfSxcbiAgICAgIH0pLFxuICAgICAgZW52aXJvbm1lbnQ6IHtcbiAgICAgICAgYnVpbGRJbWFnZTogY29kZWJ1aWxkLkxpbnV4QnVpbGRJbWFnZS5mcm9tRG9ja2VyUmVnaXN0cnkoJ2xhbWJjaS9sYW1iZGE6YnVpbGQtcHl0aG9uMy43JyksXG4gICAgICAgIGNvbXB1dGVUeXBlOiBjb2RlYnVpbGQuQ29tcHV0ZVR5cGUuTEFSR0UsXG4gICAgICAgIHByaXZpbGVnZWQ6IHRydWUsXG4gICAgICB9LFxuICAgICAgc2VjdXJpdHlHcm91cHM6IFt0aGlzLmVjMlNlY3VyaXR5R3JvdXBdLFxuICAgICAgc3VibmV0U2VsZWN0aW9uOiBwcm9wcy52cGMuc2VsZWN0U3VibmV0cyh7IHN1Ym5ldFR5cGU6IGVjMi5TdWJuZXRUeXBlLlBSSVZBVEUgfSksXG4gICAgICB0aW1lb3V0OiBjZGsuRHVyYXRpb24ubWludXRlcyg2MCksXG4gICAgfSk7XG5cbiAgICAvLyBDb25maWd1cmUgRUZTIGZvciBDb2RlQnVpbGQuXG4gICAgY29uc3QgcGFydGl0aW9uID0gcGFyZW50Lm5vZGUudHJ5R2V0Q29udGV4dCgnUGFydGl0aW9uJyk7XG4gICAgY29uc3QgY2ZuUHJvamVjdCA9IGNvZGVCdWlsZFByb2plY3Qubm9kZS5kZWZhdWx0Q2hpbGQgYXMgY29kZWJ1aWxkLkNmblByb2plY3Q7XG4gICAgY2ZuUHJvamVjdC5maWxlU3lzdGVtTG9jYXRpb25zID0gW3tcbiAgICAgIHR5cGU6IFwiRUZTXCIsXG4gICAgICBsb2NhdGlvbjogKHBhcnRpdGlvbiA9PT0gJ2F3cy1jbicpID9cbiAgICAgICAgYCR7dGhpcy5maWxlU3lzdGVtLmZpbGVTeXN0ZW1JZH0uZWZzLiR7Y2RrLkF3cy5SRUdJT059LmFtYXpvbmF3cy5jb20uY246L2AgOlxuICAgICAgICBgJHt0aGlzLmZpbGVTeXN0ZW0uZmlsZVN5c3RlbUlkfS5lZnMuJHtjZGsuQXdzLlJFR0lPTn0uYW1hem9uYXdzLmNvbTovYCxcbiAgICAgIG1vdW50UG9pbnQ6IFwiL21udC9tbFwiLFxuICAgICAgaWRlbnRpZmllcjogXCJtbFwiLFxuICAgICAgbW91bnRPcHRpb25zOiBcIm5mc3ZlcnM9NC4xLHJzaXplPTEwNDg1NzYsd3NpemU9MTA0ODU3NixoYXJkLHRpbWVvPTYwMCxyZXRyYW5zPTJcIlxuICAgIH1dO1xuICAgIGNmblByb2plY3QubG9nc0NvbmZpZyA9IHtcbiAgICAgIGNsb3VkV2F0Y2hMb2dzOiB7XG4gICAgICAgIHN0YXR1czogXCJFTkFCTEVEXCJcbiAgICAgIH1cbiAgICB9O1xuICAgIGNmblByb2plY3QuYWRkUHJvcGVydHlPdmVycmlkZShcbiAgICAgICdFbnZpcm9ubWVudC5JbWFnZVB1bGxDcmVkZW50aWFsc1R5cGUnLFxuICAgICAgJ0NPREVCVUlMRCdcbiAgICApO1xuXG4gICAgLy8gVHJpZ2dlcnMgdGhlIENvZGVCdWlsZCBwcm9qZWN0IHRvIGluc3RhbGwgdGhlIHB5dGhvbiBwYWNrYWdlcyBhbmQgbW9kZWwgdG8gdGhlIEVGUyBmaWxlIHN5c3RlbVxuICAgIGNvbnN0IHRyaWdnZXJCdWlsZFByb2plY3QgPSBuZXcgY3IuQXdzQ3VzdG9tUmVzb3VyY2UodGhpcywgJ1RyaWdnZXJDb2RlQnVpbGQnLCB7XG4gICAgICBvbkNyZWF0ZToge1xuICAgICAgICBzZXJ2aWNlOiAnQ29kZUJ1aWxkJyxcbiAgICAgICAgYWN0aW9uOiAnc3RhcnRCdWlsZCcsXG4gICAgICAgIHBhcmFtZXRlcnM6IHtcbiAgICAgICAgICBwcm9qZWN0TmFtZTogY29kZUJ1aWxkUHJvamVjdC5wcm9qZWN0TmFtZVxuICAgICAgICB9LFxuICAgICAgICBwaHlzaWNhbFJlc291cmNlSWQ6IGNyLlBoeXNpY2FsUmVzb3VyY2VJZC5mcm9tUmVzcG9uc2UoJ2J1aWxkLmlkJyksXG4gICAgICB9LFxuICAgICAgb25VcGRhdGU6IHtcbiAgICAgICAgc2VydmljZTogJ0NvZGVCdWlsZCcsXG4gICAgICAgIGFjdGlvbjogJ3N0YXJ0QnVpbGQnLFxuICAgICAgICBwYXJhbWV0ZXJzOiB7XG4gICAgICAgICAgcHJvamVjdE5hbWU6IGNvZGVCdWlsZFByb2plY3QucHJvamVjdE5hbWVcbiAgICAgICAgfSxcbiAgICAgICAgcGh5c2ljYWxSZXNvdXJjZUlkOiBjci5QaHlzaWNhbFJlc291cmNlSWQuZnJvbVJlc3BvbnNlKCdidWlsZC5pZCcpLFxuICAgICAgfSxcbiAgICAgIHBvbGljeTogY3IuQXdzQ3VzdG9tUmVzb3VyY2VQb2xpY3kuZnJvbVNka0NhbGxzKHsgcmVzb3VyY2VzOiBjci5Bd3NDdXN0b21SZXNvdXJjZVBvbGljeS5BTllfUkVTT1VSQ0UgfSlcbiAgICB9KTtcblxuICAgIC8vIENyZWF0ZSBkZXBlbmRlbmN5IGJldHdlZW4gRUZTIGFuZCBDb2RlYnVpbGRcbiAgICBjb2RlQnVpbGRQcm9qZWN0Lm5vZGUuYWRkRGVwZW5kZW5jeSh0aGlzLmFjY2Vzc1BvaW50KTtcblxuICAgIHRoaXMubGFtYmRhRnVuY3Rpb24gPSBuZXcgbGFtYmRhLkZ1bmN0aW9uKHRoaXMsIFwiTGFtYmRhRnVuY3Rpb25cIiwge1xuICAgICAgaGFuZGxlcjogXCJhcHAubGFtYmRhX2hhbmRsZXJcIixcbiAgICAgIHJ1bnRpbWU6IGxhbWJkYS5SdW50aW1lLlBZVEhPTl8zXzcsXG4gICAgICBjb2RlOiBsYW1iZGEuQ29kZS5mcm9tQXNzZXQocGF0aC5qb2luKF9fZGlybmFtZSwgJy4vc3JjLycpLCB7IC8vIFRPRE86IGRvIHdlIG5lZWQgYWxsIGZpbGVzIGluIHNyYyBmb2xkZXI/XG4gICAgICAgIGJ1bmRsaW5nOiB7XG4gICAgICAgICAgaW1hZ2U6IGxhbWJkYS5SdW50aW1lLlBZVEhPTl8zXzcuYnVuZGxpbmdEb2NrZXJJbWFnZSxcbiAgICAgICAgICBjb21tYW5kOiBbXG4gICAgICAgICAgICAnYmFzaCcsICctYycsIFtcbiAgICAgICAgICAgICAgYGNwIC1yIC9hc3NldC1pbnB1dC8qIC9hc3NldC1vdXRwdXQvYCxcbiAgICAgICAgICAgICAgYGNkIC9hc3NldC1vdXRwdXQvYCxcbiAgICAgICAgICAgICAgYGNobW9kIGEreCAuL3NldHVwLnNoYCxcbiAgICAgICAgICAgICAgYC4vc2V0dXAuc2hgLCBgbHMgLWxhaGBcbiAgICAgICAgICAgIF0uam9pbignICYmICcpXG4gICAgICAgICAgXSxcbiAgICAgICAgICB1c2VyOiAncm9vdCdcbiAgICAgICAgfVxuICAgICAgfSksXG4gICAgICBtZW1vcnlTaXplOiAyMDQ4LFxuICAgICAgdGltZW91dDogY2RrLkR1cmF0aW9uLnNlY29uZHMoMzApLFxuICAgICAgZW52aXJvbm1lbnQ6IHtcbiAgICAgICAgREFUQV9CVUNLRVQ6IHRoaXMuZGF0YUJ1Y2tldC5idWNrZXROYW1lLFxuICAgICAgICBNT0RFTF9CVUNLRVQ6IHRoaXMubW9kZWxCdWNrZXQuYnVja2V0TmFtZSxcbiAgICAgICAgUFJPRF9CVUNLRVQ6IHByb2RCdWNrZXQuYnVja2V0TmFtZSxcbiAgICAgICAgVEFTS19UQUJMRTogdGhpcy50YXNrVGFibGUudGFibGVOYW1lLFxuICAgICAgICBCQVRDSF9PUFNfUk9MRV9BUk46IGJhdGNoT3BzUm9sZS5yb2xlQXJuLFxuICAgICAgICBTQUdFTUFLRVJfRVhFQ19ST0xFX0FSTjogc2FnZW1ha2VyRXhlY1JvbGUucm9sZUFybixcbiAgICAgICAgUFJJVkFURV9TVUJORVRTOiBwcm9wcy5wcml2YXRlU3VibmV0cyxcbiAgICAgICAgVlBDX0lEOiBwcm9wcy52cGMudnBjSWQsXG4gICAgICB9LFxuICAgICAgaW5pdGlhbFBvbGljeTogW1xuICAgICAgICBzM1JlYWRQb2xpY3ksIHMzUG9saWN5LCBzM2NvbnRyb2xQb2xpY3ksIGVjc1BvbGljeSwgc2FnZW1ha2VyUG9saWN5LCBkeW5hbW9kYlBvbGljeSwgYXNnUG9saWN5XG4gICAgICBdLFxuICAgICAgc2VjdXJpdHlHcm91cDogbGFtYmRhU2VjdXJpdHlHcm91cCxcbiAgICAgIHZwYzogcHJvcHMudnBjLFxuICAgICAgZmlsZXN5c3RlbTogbGFtYmRhLkZpbGVTeXN0ZW0uZnJvbUVmc0FjY2Vzc1BvaW50KHRoaXMuYWNjZXNzUG9pbnQsIFwiL21udC9tbFwiKSxcbiAgICB9KVxuXG4gICAgdGhpcy5kYXRhQnVja2V0LmdyYW50UmVhZFdyaXRlKHRoaXMubGFtYmRhRnVuY3Rpb24pO1xuICAgIHRoaXMubW9kZWxCdWNrZXQuZ3JhbnRSZWFkV3JpdGUodGhpcy5sYW1iZGFGdW5jdGlvbik7XG4gICAgcHJvZEJ1Y2tldC5ncmFudFJlYWRXcml0ZSh0aGlzLmxhbWJkYUZ1bmN0aW9uKTtcblxuICAgIGlmIChwcm9wcy5hdXRoVHlwZSA9PT0gQXV0aFR5cGUuQ09HTklUTykge1xuICAgICAgLy8gQ3JlYXRlIENvZ25pdG8gVXNlciBQb29sXG4gICAgICB0aGlzLnVzZXJQb29sID0gbmV3IGNvZ25pdG8uVXNlclBvb2wodGhpcywgJ1VzZXJQb29sJywge1xuICAgICAgICBzZWxmU2lnblVwRW5hYmxlZDogZmFsc2UsXG4gICAgICAgIHNpZ25JbkNhc2VTZW5zaXRpdmU6IGZhbHNlLFxuICAgICAgICBzaWduSW5BbGlhc2VzOiB7XG4gICAgICAgICAgZW1haWw6IHRydWUsXG4gICAgICAgICAgdXNlcm5hbWU6IGZhbHNlLFxuICAgICAgICAgIHBob25lOiB0cnVlXG4gICAgICAgIH1cbiAgICAgIH0pXG5cbiAgICAgIC8vIENyZWF0ZSBVc2VyIFBvb2wgQ2xpZW50XG4gICAgICB0aGlzLnVzZXJQb29sQXBpQ2xpZW50ID0gbmV3IGNvZ25pdG8uVXNlclBvb2xDbGllbnQodGhpcywgJ1VzZXJQb29sQXBpQ2xpZW50Jywge1xuICAgICAgICB1c2VyUG9vbDogdGhpcy51c2VyUG9vbCxcbiAgICAgICAgdXNlclBvb2xDbGllbnROYW1lOiAnUmVwbGljYXRpb25IdWJQb3J0YWwnLFxuICAgICAgICBwcmV2ZW50VXNlckV4aXN0ZW5jZUVycm9yczogdHJ1ZVxuICAgICAgfSlcbiAgICB9XG5cbiAgICBjb25zdCBsYW1iZGFGbiA9IG5ldyBhcGlnYXRld2F5LkxhbWJkYUludGVncmF0aW9uKHRoaXMubGFtYmRhRnVuY3Rpb24sIHt9KTtcbiAgICBjb25zdCBjdXN0b21PcHRpb25zID0ge1xuICAgICAgYXV0aG9yaXphdGlvblR5cGU6IGFwaWdhdGV3YXkuQXV0aG9yaXphdGlvblR5cGUuQ1VTVE9NLFxuICAgICAgYXV0aG9yaXplcjogbmV3IGFwaWdhdGV3YXkuVG9rZW5BdXRob3JpemVyKHRoaXMsIFwiVG9rZW5BdXRob3JpemVyXCIsIHtcbiAgICAgICAgaGFuZGxlcjogdGhpcy5sYW1iZGFGdW5jdGlvbixcbiAgICAgICAgaWRlbnRpdHlTb3VyY2U6IFwibWV0aG9kLnJlcXVlc3QuaGVhZGVyLmF1dGhvcml6YXRpb25cIixcbiAgICAgICAgcmVzdWx0c0NhY2hlVHRsOiBjZGsuRHVyYXRpb24ubWludXRlcyg1KVxuICAgICAgfSlcbiAgICB9XG4gICAgY29uc3QgbWV0aG9kT3B0aW9ucyA9IGN1c3RvbU9wdGlvbnM7XG5cbiAgICBjb25zdCB0YXNrQXBpID0gdGhpcy5yZXN0QXBpLnJvb3QuYWRkUmVzb3VyY2UoJ3Rhc2tzJyk7XG4gICAgdGFza0FwaS5hZGRNZXRob2QoJ0dFVCcsIGxhbWJkYUZuLCBtZXRob2RPcHRpb25zKTtcbiAgICB0YXNrQXBpLmFkZE1ldGhvZCgnUE9TVCcsIGxhbWJkYUZuLCBtZXRob2RPcHRpb25zKTtcbiAgICBjb25zdCB0YXNrSWRBcGkgPSB0YXNrQXBpLmFkZFJlc291cmNlKCd7dGFza19pZH0nKTtcbiAgICB0YXNrSWRBcGkuYWRkTWV0aG9kKCdERUxFVEUnLCBsYW1iZGFGbiwgbWV0aG9kT3B0aW9ucyk7XG4gICAgdGFza0lkQXBpLmFkZE1ldGhvZCgnR0VUJywgbGFtYmRhRm4sIG1ldGhvZE9wdGlvbnMpO1xuICAgIGNvbnN0IHRhc2tTdG9wQXBpID0gdGFza0lkQXBpLmFkZFJlc291cmNlKCdzdG9wJyk7XG4gICAgdGFza1N0b3BBcGkuYWRkTWV0aG9kKCdQT1NUJywgbGFtYmRhRm4sIG1ldGhvZE9wdGlvbnMpO1xuICAgIGNvbnN0IHRhc2tEYXRhQXBpID0gdGFza0lkQXBpLmFkZFJlc291cmNlKCdkYXRhJyk7XG4gICAgdGFza0RhdGFBcGkuYWRkTWV0aG9kKCdHRVQnLCBsYW1iZGFGbiwgbWV0aG9kT3B0aW9ucyk7XG4gICAgY29uc3QgdGFza1N0YXR1c0FwaSA9IHRhc2tJZEFwaS5hZGRSZXNvdXJjZSgnc3RhdHVzJyk7XG4gICAgdGFza1N0YXR1c0FwaS5hZGRNZXRob2QoJ0dFVCcsIGxhbWJkYUZuLCBtZXRob2RPcHRpb25zKTtcbiAgICBjb25zdCB0YXNrRGF0YUNsYXNzSWRBcGkgPSB0YXNrRGF0YUFwaS5hZGRSZXNvdXJjZSgne2NsYXNzX2lkfScpO1xuICAgIHRhc2tEYXRhQ2xhc3NJZEFwaS5hZGRNZXRob2QoJ0dFVCcsIGxhbWJkYUZuLCBtZXRob2RPcHRpb25zKTtcbiAgICB0YXNrRGF0YUNsYXNzSWRBcGkuYWRkTWV0aG9kKCdQT1NUJywgbGFtYmRhRm4sIG1ldGhvZE9wdGlvbnMpO1xuICAgIGNvbnN0IHRhc2tTM0RhdGFBcGkgPSB0YXNrSWRBcGkuYWRkUmVzb3VyY2UoJ3MzZGF0YScpO1xuICAgIHRhc2tTM0RhdGFBcGkuYWRkTWV0aG9kKCdHRVQnLCBsYW1iZGFGbiwgbWV0aG9kT3B0aW9ucyk7XG4gICAgdGFza1MzRGF0YUFwaS5hZGRNZXRob2QoJ1BPU1QnLCBsYW1iZGFGbiwgbWV0aG9kT3B0aW9ucyk7XG4gICAgY29uc3QgdGFza1RyYWluQXBpID0gdGFza0lkQXBpLmFkZFJlc291cmNlKCd0cmFpbicpO1xuICAgIHRhc2tUcmFpbkFwaS5hZGRNZXRob2QoJ1BPU1QnLCBsYW1iZGFGbiwgbWV0aG9kT3B0aW9ucyk7XG4gICAgY29uc3QgdGFza1ByZWRpY3RBcGkgPSB0YXNrSWRBcGkuYWRkUmVzb3VyY2UoJ3ByZWRpY3QnKTtcbiAgICB0YXNrUHJlZGljdEFwaS5hZGRNZXRob2QoJ1BPU1QnLCBsYW1iZGFGbiwgbWV0aG9kT3B0aW9ucyk7XG4gICAgY29uc3QgdGFza1ByZWRpY3RWMkFwaSA9IHRhc2tJZEFwaS5hZGRSZXNvdXJjZSgncHJlZGljdF92MicpO1xuICAgIHRhc2tQcmVkaWN0VjJBcGkuYWRkTWV0aG9kKCdQT1NUJywgbGFtYmRhRm4sIG1ldGhvZE9wdGlvbnMpO1xuICAgIGNvbnN0IHRhc2tEZXBsb3lBcGkgPSB0YXNrSWRBcGkuYWRkUmVzb3VyY2UoJ2RlcGxveScpO1xuICAgIHRhc2tEZXBsb3lBcGkuYWRkTWV0aG9kKCdQT1NUJywgbGFtYmRhRm4sIG1ldGhvZE9wdGlvbnMpO1xuXG4gICAgYWRkQ29yc09wdGlvbnModGFza0FwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza0lkQXBpKTtcbiAgICBhZGRDb3JzT3B0aW9ucyh0YXNrU3RvcEFwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza1N0YXR1c0FwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza0RhdGFBcGkpO1xuICAgIGFkZENvcnNPcHRpb25zKHRhc2tEYXRhQ2xhc3NJZEFwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza1MzRGF0YUFwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza1RyYWluQXBpKTtcbiAgICBhZGRDb3JzT3B0aW9ucyh0YXNrUHJlZGljdEFwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza1ByZWRpY3RWMkFwaSk7XG4gICAgYWRkQ29yc09wdGlvbnModGFza0RlcGxveUFwaSk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGFkZENvcnNPcHRpb25zKGFwaVJlc291cmNlOiBhcGlnYXRld2F5LklSZXNvdXJjZSkge1xuICBhcGlSZXNvdXJjZS5hZGRNZXRob2QoJ09QVElPTlMnLCBuZXcgYXBpZ2F0ZXdheS5Nb2NrSW50ZWdyYXRpb24oe1xuICAgIGludGVncmF0aW9uUmVzcG9uc2VzOiBbe1xuICAgICAgc3RhdHVzQ29kZTogJzIwMCcsXG4gICAgICByZXNwb25zZVBhcmFtZXRlcnM6IHtcbiAgICAgICAgJ21ldGhvZC5yZXNwb25zZS5oZWFkZXIuQWNjZXNzLUNvbnRyb2wtQWxsb3ctSGVhZGVycyc6IFwiJ0NvbnRlbnQtVHlwZSxYLUFtei1EYXRlLEF1dGhvcml6YXRpb24sWC1BcGktS2V5LFgtQW16LVNlY3VyaXR5LVRva2VuLFgtQW16LVVzZXItQWdlbnQnXCIsXG4gICAgICAgICdtZXRob2QucmVzcG9uc2UuaGVhZGVyLkFjY2Vzcy1Db250cm9sLUFsbG93LU9yaWdpbic6IFwiJyonXCIsXG4gICAgICAgICdtZXRob2QucmVzcG9uc2UuaGVhZGVyLkFjY2Vzcy1Db250cm9sLUFsbG93LUNyZWRlbnRpYWxzJzogXCInZmFsc2UnXCIsXG4gICAgICAgICdtZXRob2QucmVzcG9uc2UuaGVhZGVyLkFjY2Vzcy1Db250cm9sLUFsbG93LU1ldGhvZHMnOiBcIidPUFRJT05TLEdFVCxQVVQsUE9TVCxERUxFVEUnXCIsXG4gICAgICB9LFxuICAgIH1dLFxuICAgIHBhc3N0aHJvdWdoQmVoYXZpb3I6IGFwaWdhdGV3YXkuUGFzc3Rocm91Z2hCZWhhdmlvci5ORVZFUixcbiAgICByZXF1ZXN0VGVtcGxhdGVzOiB7XG4gICAgICBcImFwcGxpY2F0aW9uL2pzb25cIjogXCJ7XFxcInN0YXR1c0NvZGVcXFwiOiAyMDB9XCJcbiAgICB9LFxuICB9KSwge1xuICAgIG1ldGhvZFJlc3BvbnNlczogW3tcbiAgICAgIHN0YXR1c0NvZGU6ICcyMDAnLFxuICAgICAgcmVzcG9uc2VQYXJhbWV0ZXJzOiB7XG4gICAgICAgICdtZXRob2QucmVzcG9uc2UuaGVhZGVyLkFjY2Vzcy1Db250cm9sLUFsbG93LUhlYWRlcnMnOiB0cnVlLFxuICAgICAgICAnbWV0aG9kLnJlc3BvbnNlLmhlYWRlci5BY2Nlc3MtQ29udHJvbC1BbGxvdy1NZXRob2RzJzogdHJ1ZSxcbiAgICAgICAgJ21ldGhvZC5yZXNwb25zZS5oZWFkZXIuQWNjZXNzLUNvbnRyb2wtQWxsb3ctQ3JlZGVudGlhbHMnOiB0cnVlLFxuICAgICAgICAnbWV0aG9kLnJlc3BvbnNlLmhlYWRlci5BY2Nlc3MtQ29udHJvbC1BbGxvdy1PcmlnaW4nOiB0cnVlLFxuICAgICAgfSxcbiAgICB9XSxcbiAgfSlcbn1cbiJdfQ==