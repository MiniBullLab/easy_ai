import * as iam from "@aws-cdk/aws-iam"
import * as lambda from "@aws-cdk/aws-lambda";
import {Vpc} from "@aws-cdk/aws-ec2"
import * as cdk from "@aws-cdk/core";
import * as apigateway from '@aws-cdk/aws-apigateway';
import * as dynamodb from '@aws-cdk/aws-dynamodb';
import * as path from 'path';
import * as s3 from '@aws-cdk/aws-s3';
import * as cognito from '@aws-cdk/aws-cognito';
import * as efs from '@aws-cdk/aws-efs';
import * as ec2 from '@aws-cdk/aws-ec2';
import * as codebuild from "@aws-cdk/aws-codebuild"
import * as cr from "@aws-cdk/custom-resources"
import { AuthType } from './index';

export interface LambdaApiStackProps {
  readonly authType: AuthType;
  readonly vpc: ec2.IVpc;
  readonly privateSubnets: string;
}

export class LambdaApiStack extends cdk.Construct {
  readonly restApi: apigateway.RestApi
  readonly lambdaFunction: lambda.Function
  readonly dataBucket: s3.Bucket
  readonly modelBucket: s3.Bucket
  // readonly vpc: Vpc
  readonly taskTable: dynamodb.Table

  readonly userPool?: cognito.UserPool
  readonly userPoolApiClient?: cognito.UserPoolClient
  readonly userPoolDomain?: cognito.UserPoolDomain

  readonly fileSystem : efs.FileSystem
  readonly accessPoint : efs.AccessPoint
  readonly ec2SecurityGroup : ec2.SecurityGroup

  constructor(parent: cdk.Construct, id: string, props: LambdaApiStackProps) {
    super(parent, id)

    this.taskTable = new dynamodb.Table(this, 'TaskTable', {
      partitionKey: { name: 'JobId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY
    });
    
    // this.vpc = new Vpc(this, 'Vpc', { maxAzs: 2, natGateways: 1 });
    this.dataBucket = new s3.Bucket(this, "DataBucket", {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true
    })
    this.modelBucket = new s3.Bucket(this, "ModelBucket", {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true
    })
    const prodBucket = new s3.Bucket(this, "ProdBucket", {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: false
    })

    this.restApi = new apigateway.RestApi(this, "RestApi", {
      restApiName: cdk.Aws.STACK_NAME,
      deployOptions: {
        stageName: "Prod",
        metricsEnabled: true,
        loggingLevel: apigateway.MethodLoggingLevel.INFO,
        dataTraceEnabled: true,
      },
      endpointConfiguration: {
        types: [ apigateway.EndpointType.REGIONAL ]
      }
    })

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
    
    const s3ReadPolicy = new iam.PolicyStatement()
    s3ReadPolicy.addActions("s3:GetObject*")
    s3ReadPolicy.addActions("s3:ListBucket")
    s3ReadPolicy.addResources("*")
    const s3Policy = new iam.PolicyStatement()
    s3Policy.addActions("s3:*")
    s3Policy.addResources(this.dataBucket.bucketArn + "/*")
    s3Policy.addResources(this.modelBucket.bucketArn + "/*")
    s3Policy.addResources(prodBucket.bucketArn + "/*")
    const s3controlPolicy = new iam.PolicyStatement()
    s3controlPolicy.addActions("s3:*")
    s3controlPolicy.addActions("iam:*")
    s3controlPolicy.addResources("*")
    const ecsPolicy = new iam.PolicyStatement()
    ecsPolicy.addActions("ec2:*")
    ecsPolicy.addActions("ecs:*")
    ecsPolicy.addActions("iam:*")
    ecsPolicy.addResources("*")
    const sagemakerPolicy = new iam.PolicyStatement()
    sagemakerPolicy.addActions("sagemaker:*")
    sagemakerPolicy.addResources("*")
    const dynamodbPolicy = new iam.PolicyStatement()
    dynamodbPolicy.addActions("dynamodb:*")
    dynamodbPolicy.addActions("iam:*")
    dynamodbPolicy.addResources(this.taskTable.tableArn)
    const asgPolicy = new iam.PolicyStatement()
    asgPolicy.addActions("autoscaling:*")
    asgPolicy.addActions("elasticloadbalancing:*")
    asgPolicy.addActions("cloudwatch:*")
    asgPolicy.addResources("*")

    const batchOpsRole = new iam.Role(this, "AdminRoleForS3BatchOperations", {
      assumedBy: new iam.ServicePrincipal('batchoperations.s3.amazonaws.com'),
      description: "Allows S3 Batch Operations to call AWS services on your behalf.",
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonS3FullAccess")
      ]
    })

    const sagemakerExecRole = new iam.Role(this, "AdminRoleForSageMakerExecution", {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      description: "Allows SageMaker Endpoints to call AWS services on your behalf.",
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonS3FullAccess"),
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonEC2ContainerRegistryFullAccess"),
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess"),
      ]
    })

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
      path: '/',            // remove EFS access point
      posixUser: {
        gid: '0',
        uid: '0'
      },
      createAcl: {
        ownerGid: '0',
        ownerUid: '0',
        permissions: '777'
      }
    })

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
              '/mnt/ml/code/bin/pip3 install -i https://opentuna.cn/pypi/web/simple opencv-python==4.4.0.44',
              '/mnt/ml/code/bin/pip3 install -i https://opentuna.cn/pypi/web/simple pytorch-cpu==1.0.1 torchvision-cpu==0.2.2',
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
    const cfnProject = codeBuildProject.node.defaultChild as codebuild.CfnProject;
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
    cfnProject.addPropertyOverride(
      'Environment.ImagePullCredentialsType',
      'CODEBUILD'
    );

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
      code: lambda.Code.fromAsset(path.join(__dirname, './src/'), { // TODO: do we need all files in src folder?
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
    })

    this.dataBucket.grantReadWrite(this.lambdaFunction);
    this.modelBucket.grantReadWrite(this.lambdaFunction);
    prodBucket.grantReadWrite(this.lambdaFunction);

    if (props.authType === AuthType.COGNITO) {
      // Create Cognito User Pool
      this.userPool = new cognito.UserPool(this, 'UserPool', {
        selfSignUpEnabled: false,
        signInCaseSensitive: false,
        signInAliases: {
          email: true,
          username: false,
          phone: true
        }
      })

      // Create User Pool Client
      this.userPoolApiClient = new cognito.UserPoolClient(this, 'UserPoolApiClient', {
        userPool: this.userPool,
        userPoolClientName: 'ReplicationHubPortal',
        preventUserExistenceErrors: true
      })
    }

    const lambdaFn = new apigateway.LambdaIntegration(this.lambdaFunction, {});
    const customOptions = {
      authorizationType: apigateway.AuthorizationType.CUSTOM,
      authorizer: new apigateway.TokenAuthorizer(this, "TokenAuthorizer", {
        handler: this.lambdaFunction,
        identitySource: "method.request.header.authorization",
        resultsCacheTtl: cdk.Duration.minutes(5)
      })
    }
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

export function addCorsOptions(apiResource: apigateway.IResource) {
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
  })
}
