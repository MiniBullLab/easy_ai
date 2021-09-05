import { StaticSite } from './static-site';
import { LambdaApiStack } from "./lambda-api-stack"
import { EcsStack } from "./ecs-stack"
import { Rule } from '@aws-cdk/aws-events';
import { LambdaFunction } from '@aws-cdk/aws-events-targets';
import * as ec2 from '@aws-cdk/aws-ec2';
import * as cdk from "@aws-cdk/core";

export const enum AuthType {
  COGNITO = "cognito",
  OPENID = "openid"
}

interface MLBotSettings {
  vpc?: ec2.IVpc;
  // publicSubnets: ec2.SubnetSelection;
  privateSubnets?: ec2.SubnetSelection;
}

class MLBotStack extends cdk.Stack {
  private _paramGroup: { [grpname: string]: cdk.CfnParameter[]} = {}
  private _mlbotSettings: MLBotSettings = { };

  protected makeParam(id: string, props?: cdk.CfnParameterProps): cdk.CfnParameter { return new cdk.CfnParameter(this, id, props); }
  protected addGroupParam(props: { [key: string]: cdk.CfnParameter[]}): void {
    for (const key of Object.keys(props)) {
      const params = props[key];
      this._paramGroup[key] = params.concat(this._paramGroup[key] ?? []);
    }
    this._setParamGroups();
  }
  private _setParamGroups(): void {
    if (!this.templateOptions.metadata) { this.templateOptions.metadata = {}; }
    const mkgrp = (label: string, params: cdk.CfnParameter[]) => {
      return {
        Label: { default: label },
        Parameters: params.map(p => {
          return p ? p.logicalId : '';
        }).filter(id => id),
      };
    };
    this.templateOptions.metadata['AWS::CloudFormation::Interface'] = {
      ParameterGroups: Object.keys(this._paramGroup).map(key => mkgrp(key, this._paramGroup[key]) ),
    };
  }

  constructor(parent: cdk.App, name: string, props?: cdk.StackProps) {
    super(parent, name, props);

    const version = (process.env.VERSION && process.env.VERSION.startsWith('v')) ? process.env.VERSION : 'v1.1.2'
    
    const partition = parent.node.tryGetContext('Partition');
    const existingVpc = parent.node.tryGetContext('ExistingVPC');
    const vpcMsg = existingVpc ? 'existing vpc' : 'new vpc';
    this.templateOptions.description = `(SO8012) Machine Learning Bot with ${vpcMsg}. Template version ${version}`
    let authType;
    let domain, loginDomain; // sslCertificateId,
    let vpc;

    if (partition === 'aws-cn') {
      authType = AuthType.OPENID;
    } else {
      authType = AuthType.COGNITO;
    }

    if (partition === 'aws-cn') {
      domain = new cdk.CfnParameter(this, 'Domain', {
        description: 'Domain Name',
        type: 'String',
        allowedPattern: '(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\\.)+[a-z0-9][a-z0-9-]{0,61}[a-z]'
      });

      loginDomain = new cdk.CfnParameter(this, 'LoginDomain', {
        type: 'String',
        allowedPattern: '(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\\.)+[a-z0-9][a-z0-9-]{0,61}[a-z]',
        description: 'The domain address for OIDC based login, e.g. ml-bot.authing.cn'
      })

      this.templateOptions.metadata = {
        'AWS::CloudFormation::Interface': {
          ParameterGroups: [
            {
              Label: { default: 'Portal' },
              Parameters: [ domain.logicalId, loginDomain.logicalId ] // sslCertificateId.logicalId, 
            }
          ],
          ParameterLabels: {
            // SSLCertificateId: { default: 'SSL Certificate ID' },
            Domain: { default: 'Domain' },
            LoginDomain: { default: 'Login Domain' }
          }
        }
      }
    }

    if (existingVpc == "true") {
      const vpcIdParam = this.makeParam('VpcId', {
        type: 'AWS::EC2::VPC::Id',
        description: 'Your VPC Id',
      });
      // const pubSubnetsParam = this.makeParam('PubSubnets', {
      //   type: 'List<AWS::EC2::Subnet::Id>',
      //   description: 'Public subnets (Choose two)',
      // });
      const privSubnetsParam = this.makeParam('PrivSubnets', {
        type: 'List<AWS::EC2::Subnet::Id>',
        description: 'Private subnets (Choose two)',
      });
      this.addGroupParam({ 'VPC Settings': [vpcIdParam, privSubnetsParam] }); // pubSubnetsParam, 

      const azs = ['a', 'b'];
      vpc = ec2.Vpc.fromVpcAttributes(this, 'VpcAttr', {
        vpcId: vpcIdParam.valueAsString,
        vpcCidrBlock: cdk.Aws.NO_VALUE,
        availabilityZones: azs,
        // publicSubnetIds: azs.map((_, index) => Fn.select(index, pubSubnetsParam.valueAsList)),
        privateSubnetIds: azs.map((_, index) => cdk.Fn.select(index, privSubnetsParam.valueAsList)),
      });

      Object.assign(this._mlbotSettings, {
        vpc,
        // publicSubnets: { subnets: vpc.publicSubnets },
        privateSubnets: { subnets: vpc.privateSubnets },
      });
    } else {
      vpc = new ec2.Vpc(this, 'Vpc', { maxAzs: 2, natGateways: 1 });
      Object.assign(this._mlbotSettings, {
        vpc,
        // publicSubnets: { subnets: vpc.publicSubnets },
        privateSubnets: { subnets: vpc.privateSubnets },
      });
    }

    const privateSubnets = vpc.privateSubnets[2] ? 
      vpc.privateSubnets[0].subnetId + "," + vpc.privateSubnets[1].subnetId + "," + vpc.privateSubnets[2].subnetId :
      vpc.privateSubnets[0].subnetId + "," + vpc.privateSubnets[1].subnetId;

    const lambdaApi = new LambdaApiStack(this, 'LambdaApi', {
      authType: authType,
      vpc: vpc,
      privateSubnets: privateSubnets,
    });

    const ecsStack = new EcsStack(this, 'EcsStack', {
      dataBucket: lambdaApi.dataBucket,
      modelBucket: lambdaApi.modelBucket,
      vpc: vpc,
      version: version,
      fileSystemId: lambdaApi.fileSystem.fileSystemId,
      accessPointId: lambdaApi.accessPoint.accessPointId,
      ec2SecurityGroup : lambdaApi.ec2SecurityGroup,
    });

    lambdaApi.lambdaFunction.addEnvironment("CLUSTER_ARN", ecsStack.cluster.clusterArn)
    lambdaApi.lambdaFunction.addEnvironment("TRAINING_TASK_ARN", ecsStack.trainingTaskDef.taskDefinitionArn)
    lambdaApi.lambdaFunction.addEnvironment("INFERENCE_TASK_ARN", ecsStack.inferenceTaskDef.taskDefinitionArn)
    lambdaApi.lambdaFunction.addEnvironment("INFERENCE_FAMILY", ecsStack.inferenceTaskDef.family)
    lambdaApi.lambdaFunction.addEnvironment("TRAINING_ASG", ecsStack.trainingAsgName)
    lambdaApi.lambdaFunction.addEnvironment("INFERENCE_ASG", ecsStack.inferenceAsgName)
    lambdaApi.lambdaFunction.addEnvironment("INFERENCE_SG", ecsStack.inferenceSG.securityGroupId)
    if (loginDomain && partition === 'aws-cn') {
      lambdaApi.lambdaFunction.addEnvironment("LOGIN_DOMAIN", `https://${loginDomain.valueAsString}/oidc`)
    } else if (lambdaApi.userPool) {
      lambdaApi.lambdaFunction.addEnvironment("LOGIN_DOMAIN", lambdaApi.userPool.userPoolProviderUrl)
    }

    /**
     * Create a EventBridge rule to automate model deployment process using Lambda
     */
    const eventBridgeTarget = new LambdaFunction(lambdaApi.lambdaFunction);
    const ecsEventPattern = {
      source: [ "aws.ecs" ],
      detailType: [ "ECS Task State Change" ],
      detail: { "clusterArn": [ ecsStack.cluster.clusterArn ] }
    }
    new Rule(this, 'EcsEventPattern', {
      eventPattern: ecsEventPattern,
      targets: [eventBridgeTarget],
    });
    const autoscalingEventPattern = {
      source: [ "aws.autoscaling" ],
      detailType: [
        "EC2 Instance Launch Successful",
        "EC2 Instance Terminate Successful",
        "EC2 Instance Launch Unsuccessful",
        "EC2 Instance Terminate Unsuccessful",
        "EC2 Instance-launch Lifecycle Action",
        "EC2 Instance-terminate Lifecycle Action"
      ],
    }
    new Rule(this, 'AutoscalingEventPattern', {
      eventPattern: autoscalingEventPattern,
      targets: [eventBridgeTarget],
    });
    
    const portal = new StaticSite(this, 'StaticSite', {
      ...(domain) && {domain: domain.valueAsString},
      ...(loginDomain) && {loginDomain: loginDomain.valueAsString},
      apiUrl: lambdaApi.restApi.url,
      auth_type: authType,
      aws_user_pools_id: lambdaApi.userPool?.userPoolId || '',
      aws_user_pools_web_client_id: lambdaApi.userPoolApiClient?.userPoolClientId || '',
    });

    new cdk.CfnOutput(this, 'DistributionIdOutput', {
      value: portal.website.distributionId,
      description: 'The Portal CloudFront Distribution ID.'
    }).overrideLogicalId('DistributionId');
    new cdk.CfnOutput(this, 'DistributionDomainOutput', {
      value: portal.website.distributionDomainName,
      description: 'The domain name of the portal. Create a CNAME and point to this record.'
    }).overrideLogicalId('CloudFrontDomain');
    if (domain) {
      new cdk.CfnOutput(this, 'DomainOutput', {
        value: domain.valueAsString,
        description: 'The domain URL of the portal'
      }).overrideLogicalId('Domain')
    }
    if (lambdaApi.userPool) {
      new cdk.CfnOutput(this, 'UserPoolIdOutput', {
        value: lambdaApi.userPool.userPoolId,
        description: 'The Cognito UserPool for managing users of the portal'
      }).overrideLogicalId('UserPoolId')
    }

  }
}

const app = new cdk.App();
new MLBotStack(app, 'ml-bot');

app.synth();
