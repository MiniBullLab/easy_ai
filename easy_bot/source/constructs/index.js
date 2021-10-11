"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const static_site_1 = require("./static-site");
const lambda_api_stack_1 = require("./lambda-api-stack");
const ecs_stack_1 = require("./ecs-stack");
const aws_events_1 = require("@aws-cdk/aws-events");
const aws_events_targets_1 = require("@aws-cdk/aws-events-targets");
const ec2 = require("@aws-cdk/aws-ec2");
const cdk = require("@aws-cdk/core");
class MLBotStack extends cdk.Stack {
    constructor(parent, name, props) {
        var _a, _b;
        super(parent, name, props);
        this._paramGroup = {};
        this._mlbotSettings = {};
        const version = (process.env.VERSION && process.env.VERSION.startsWith('v')) ? process.env.VERSION : 'v1.1.2';
        const partition = parent.node.tryGetContext('Partition');
        const existingVpc = parent.node.tryGetContext('ExistingVPC');
        const vpcMsg = existingVpc ? 'existing vpc' : 'new vpc';
        this.templateOptions.description = `(SO8012) Machine Learning Bot with ${vpcMsg}. Template version ${version}`;
        let authType;
        let domain, loginDomain; // sslCertificateId,
        let vpc;
        if (partition === 'aws-cn') {
            authType = "openid" /* OPENID */;
        }
        else {
            authType = "cognito" /* COGNITO */;
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
            });
            this.templateOptions.metadata = {
                'AWS::CloudFormation::Interface': {
                    ParameterGroups: [
                        {
                            Label: { default: 'Portal' },
                            Parameters: [domain.logicalId, loginDomain.logicalId] // sslCertificateId.logicalId, 
                        }
                    ],
                    ParameterLabels: {
                        // SSLCertificateId: { default: 'SSL Certificate ID' },
                        Domain: { default: 'Domain' },
                        LoginDomain: { default: 'Login Domain' }
                    }
                }
            };
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
        }
        else {
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
        const lambdaApi = new lambda_api_stack_1.LambdaApiStack(this, 'LambdaApi', {
            authType: authType,
            vpc: vpc,
            privateSubnets: privateSubnets,
        });
        const ecsStack = new ecs_stack_1.EcsStack(this, 'EcsStack', {
            dataBucket: lambdaApi.dataBucket,
            modelBucket: lambdaApi.modelBucket,
            vpc: vpc,
            version: version,
            fileSystemId: lambdaApi.fileSystem.fileSystemId,
            accessPointId: lambdaApi.accessPoint.accessPointId,
            ec2SecurityGroup: lambdaApi.ec2SecurityGroup,
        });
        lambdaApi.lambdaFunction.addEnvironment("CLUSTER_ARN", ecsStack.cluster.clusterArn);
        lambdaApi.lambdaFunction.addEnvironment("TRAINING_TASK_ARN", ecsStack.trainingTaskDef.taskDefinitionArn);
        lambdaApi.lambdaFunction.addEnvironment("INFERENCE_TASK_ARN", ecsStack.inferenceTaskDef.taskDefinitionArn);
        lambdaApi.lambdaFunction.addEnvironment("INFERENCE_FAMILY", ecsStack.inferenceTaskDef.family);
        lambdaApi.lambdaFunction.addEnvironment("TRAINING_ASG", ecsStack.trainingAsgName);
        lambdaApi.lambdaFunction.addEnvironment("INFERENCE_ASG", ecsStack.inferenceAsgName);
        lambdaApi.lambdaFunction.addEnvironment("INFERENCE_SG", ecsStack.inferenceSG.securityGroupId);
        if (loginDomain && partition === 'aws-cn') {
            lambdaApi.lambdaFunction.addEnvironment("LOGIN_DOMAIN", `https://${loginDomain.valueAsString}/oidc`);
        }
        else if (lambdaApi.userPool) {
            lambdaApi.lambdaFunction.addEnvironment("LOGIN_DOMAIN", lambdaApi.userPool.userPoolProviderUrl);
        }
        /**
         * Create a EventBridge rule to automate model deployment process using Lambda
         */
        const eventBridgeTarget = new aws_events_targets_1.LambdaFunction(lambdaApi.lambdaFunction);
        const ecsEventPattern = {
            source: ["aws.ecs"],
            detailType: ["ECS Task State Change"],
            detail: { "clusterArn": [ecsStack.cluster.clusterArn] }
        };
        new aws_events_1.Rule(this, 'EcsEventPattern', {
            eventPattern: ecsEventPattern,
            targets: [eventBridgeTarget],
        });
        const autoscalingEventPattern = {
            source: ["aws.autoscaling"],
            detailType: [
                "EC2 Instance Launch Successful",
                "EC2 Instance Terminate Successful",
                "EC2 Instance Launch Unsuccessful",
                "EC2 Instance Terminate Unsuccessful",
                "EC2 Instance-launch Lifecycle Action",
                "EC2 Instance-terminate Lifecycle Action"
            ],
        };
        new aws_events_1.Rule(this, 'AutoscalingEventPattern', {
            eventPattern: autoscalingEventPattern,
            targets: [eventBridgeTarget],
        });
        const portal = new static_site_1.StaticSite(this, 'StaticSite', {
            ...(domain) && { domain: domain.valueAsString },
            ...(loginDomain) && { loginDomain: loginDomain.valueAsString },
            apiUrl: lambdaApi.restApi.url,
            auth_type: authType,
            aws_user_pools_id: ((_a = lambdaApi.userPool) === null || _a === void 0 ? void 0 : _a.userPoolId) || '',
            aws_user_pools_web_client_id: ((_b = lambdaApi.userPoolApiClient) === null || _b === void 0 ? void 0 : _b.userPoolClientId) || '',
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
            }).overrideLogicalId('Domain');
        }
        if (lambdaApi.userPool) {
            new cdk.CfnOutput(this, 'UserPoolIdOutput', {
                value: lambdaApi.userPool.userPoolId,
                description: 'The Cognito UserPool for managing users of the portal'
            }).overrideLogicalId('UserPoolId');
        }
    }
    makeParam(id, props) { return new cdk.CfnParameter(this, id, props); }
    addGroupParam(props) {
        var _a;
        for (const key of Object.keys(props)) {
            const params = props[key];
            this._paramGroup[key] = params.concat((_a = this._paramGroup[key]) !== null && _a !== void 0 ? _a : []);
        }
        this._setParamGroups();
    }
    _setParamGroups() {
        if (!this.templateOptions.metadata) {
            this.templateOptions.metadata = {};
        }
        const mkgrp = (label, params) => {
            return {
                Label: { default: label },
                Parameters: params.map(p => {
                    return p ? p.logicalId : '';
                }).filter(id => id),
            };
        };
        this.templateOptions.metadata['AWS::CloudFormation::Interface'] = {
            ParameterGroups: Object.keys(this._paramGroup).map(key => mkgrp(key, this._paramGroup[key])),
        };
    }
}
const app = new cdk.App();
new MLBotStack(app, 'ml-bot');
app.synth();
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW5kZXguanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyJpbmRleC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOztBQUFBLCtDQUEyQztBQUMzQyx5REFBbUQ7QUFDbkQsMkNBQXNDO0FBQ3RDLG9EQUEyQztBQUMzQyxvRUFBNkQ7QUFDN0Qsd0NBQXdDO0FBQ3hDLHFDQUFxQztBQWFyQyxNQUFNLFVBQVcsU0FBUSxHQUFHLENBQUMsS0FBSztJQTJCaEMsWUFBWSxNQUFlLEVBQUUsSUFBWSxFQUFFLEtBQXNCOztRQUMvRCxLQUFLLENBQUMsTUFBTSxFQUFFLElBQUksRUFBRSxLQUFLLENBQUMsQ0FBQztRQTNCckIsZ0JBQVcsR0FBNkMsRUFBRSxDQUFBO1FBQzFELG1CQUFjLEdBQWtCLEVBQUcsQ0FBQztRQTRCMUMsTUFBTSxPQUFPLEdBQUcsQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLE9BQU8sSUFBSSxPQUFPLENBQUMsR0FBRyxDQUFDLE9BQU8sQ0FBQyxVQUFVLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLEdBQUcsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLFFBQVEsQ0FBQTtRQUU3RyxNQUFNLFNBQVMsR0FBRyxNQUFNLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUN6RCxNQUFNLFdBQVcsR0FBRyxNQUFNLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUM3RCxNQUFNLE1BQU0sR0FBRyxXQUFXLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUMsU0FBUyxDQUFDO1FBQ3hELElBQUksQ0FBQyxlQUFlLENBQUMsV0FBVyxHQUFHLHNDQUFzQyxNQUFNLHNCQUFzQixPQUFPLEVBQUUsQ0FBQTtRQUM5RyxJQUFJLFFBQVEsQ0FBQztRQUNiLElBQUksTUFBTSxFQUFFLFdBQVcsQ0FBQyxDQUFDLG9CQUFvQjtRQUM3QyxJQUFJLEdBQUcsQ0FBQztRQUVSLElBQUksU0FBUyxLQUFLLFFBQVEsRUFBRTtZQUMxQixRQUFRLHdCQUFrQixDQUFDO1NBQzVCO2FBQU07WUFDTCxRQUFRLDBCQUFtQixDQUFDO1NBQzdCO1FBRUQsSUFBSSxTQUFTLEtBQUssUUFBUSxFQUFFO1lBQzFCLE1BQU0sR0FBRyxJQUFJLEdBQUcsQ0FBQyxZQUFZLENBQUMsSUFBSSxFQUFFLFFBQVEsRUFBRTtnQkFDNUMsV0FBVyxFQUFFLGFBQWE7Z0JBQzFCLElBQUksRUFBRSxRQUFRO2dCQUNkLGNBQWMsRUFBRSwwRUFBMEU7YUFDM0YsQ0FBQyxDQUFDO1lBRUgsV0FBVyxHQUFHLElBQUksR0FBRyxDQUFDLFlBQVksQ0FBQyxJQUFJLEVBQUUsYUFBYSxFQUFFO2dCQUN0RCxJQUFJLEVBQUUsUUFBUTtnQkFDZCxjQUFjLEVBQUUsMEVBQTBFO2dCQUMxRixXQUFXLEVBQUUsaUVBQWlFO2FBQy9FLENBQUMsQ0FBQTtZQUVGLElBQUksQ0FBQyxlQUFlLENBQUMsUUFBUSxHQUFHO2dCQUM5QixnQ0FBZ0MsRUFBRTtvQkFDaEMsZUFBZSxFQUFFO3dCQUNmOzRCQUNFLEtBQUssRUFBRSxFQUFFLE9BQU8sRUFBRSxRQUFRLEVBQUU7NEJBQzVCLFVBQVUsRUFBRSxDQUFFLE1BQU0sQ0FBQyxTQUFTLEVBQUUsV0FBVyxDQUFDLFNBQVMsQ0FBRSxDQUFDLCtCQUErQjt5QkFDeEY7cUJBQ0Y7b0JBQ0QsZUFBZSxFQUFFO3dCQUNmLHVEQUF1RDt3QkFDdkQsTUFBTSxFQUFFLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBRTt3QkFDN0IsV0FBVyxFQUFFLEVBQUUsT0FBTyxFQUFFLGNBQWMsRUFBRTtxQkFDekM7aUJBQ0Y7YUFDRixDQUFBO1NBQ0Y7UUFFRCxJQUFJLFdBQVcsSUFBSSxNQUFNLEVBQUU7WUFDekIsTUFBTSxVQUFVLEdBQUcsSUFBSSxDQUFDLFNBQVMsQ0FBQyxPQUFPLEVBQUU7Z0JBQ3pDLElBQUksRUFBRSxtQkFBbUI7Z0JBQ3pCLFdBQVcsRUFBRSxhQUFhO2FBQzNCLENBQUMsQ0FBQztZQUNILHlEQUF5RDtZQUN6RCx3Q0FBd0M7WUFDeEMsZ0RBQWdEO1lBQ2hELE1BQU07WUFDTixNQUFNLGdCQUFnQixHQUFHLElBQUksQ0FBQyxTQUFTLENBQUMsYUFBYSxFQUFFO2dCQUNyRCxJQUFJLEVBQUUsNEJBQTRCO2dCQUNsQyxXQUFXLEVBQUUsOEJBQThCO2FBQzVDLENBQUMsQ0FBQztZQUNILElBQUksQ0FBQyxhQUFhLENBQUMsRUFBRSxjQUFjLEVBQUUsQ0FBQyxVQUFVLEVBQUUsZ0JBQWdCLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxvQkFBb0I7WUFFNUYsTUFBTSxHQUFHLEdBQUcsQ0FBQyxHQUFHLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFDdkIsR0FBRyxHQUFHLEdBQUcsQ0FBQyxHQUFHLENBQUMsaUJBQWlCLENBQUMsSUFBSSxFQUFFLFNBQVMsRUFBRTtnQkFDL0MsS0FBSyxFQUFFLFVBQVUsQ0FBQyxhQUFhO2dCQUMvQixZQUFZLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxRQUFRO2dCQUM5QixpQkFBaUIsRUFBRSxHQUFHO2dCQUN0Qix5RkFBeUY7Z0JBQ3pGLGdCQUFnQixFQUFFLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUUsS0FBSyxFQUFFLEVBQUUsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLE1BQU0sQ0FBQyxLQUFLLEVBQUUsZ0JBQWdCLENBQUMsV0FBVyxDQUFDLENBQUM7YUFDNUYsQ0FBQyxDQUFDO1lBRUgsTUFBTSxDQUFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsY0FBYyxFQUFFO2dCQUNqQyxHQUFHO2dCQUNILGlEQUFpRDtnQkFDakQsY0FBYyxFQUFFLEVBQUUsT0FBTyxFQUFFLEdBQUcsQ0FBQyxjQUFjLEVBQUU7YUFDaEQsQ0FBQyxDQUFDO1NBQ0o7YUFBTTtZQUNMLEdBQUcsR0FBRyxJQUFJLEdBQUcsQ0FBQyxHQUFHLENBQUMsSUFBSSxFQUFFLEtBQUssRUFBRSxFQUFFLE1BQU0sRUFBRSxDQUFDLEVBQUUsV0FBVyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUM7WUFDOUQsTUFBTSxDQUFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsY0FBYyxFQUFFO2dCQUNqQyxHQUFHO2dCQUNILGlEQUFpRDtnQkFDakQsY0FBYyxFQUFFLEVBQUUsT0FBTyxFQUFFLEdBQUcsQ0FBQyxjQUFjLEVBQUU7YUFDaEQsQ0FBQyxDQUFDO1NBQ0o7UUFFRCxNQUFNLGNBQWMsR0FBRyxHQUFHLENBQUMsY0FBYyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDNUMsR0FBRyxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxRQUFRLEdBQUcsR0FBRyxHQUFHLEdBQUcsQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLENBQUMsUUFBUSxHQUFHLEdBQUcsR0FBRyxHQUFHLENBQUMsY0FBYyxDQUFDLENBQUMsQ0FBQyxDQUFDLFFBQVEsQ0FBQyxDQUFDO1lBQzlHLEdBQUcsQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLENBQUMsUUFBUSxHQUFHLEdBQUcsR0FBRyxHQUFHLENBQUMsY0FBYyxDQUFDLENBQUMsQ0FBQyxDQUFDLFFBQVEsQ0FBQztRQUV4RSxNQUFNLFNBQVMsR0FBRyxJQUFJLGlDQUFjLENBQUMsSUFBSSxFQUFFLFdBQVcsRUFBRTtZQUN0RCxRQUFRLEVBQUUsUUFBUTtZQUNsQixHQUFHLEVBQUUsR0FBRztZQUNSLGNBQWMsRUFBRSxjQUFjO1NBQy9CLENBQUMsQ0FBQztRQUVILE1BQU0sUUFBUSxHQUFHLElBQUksb0JBQVEsQ0FBQyxJQUFJLEVBQUUsVUFBVSxFQUFFO1lBQzlDLFVBQVUsRUFBRSxTQUFTLENBQUMsVUFBVTtZQUNoQyxXQUFXLEVBQUUsU0FBUyxDQUFDLFdBQVc7WUFDbEMsR0FBRyxFQUFFLEdBQUc7WUFDUixPQUFPLEVBQUUsT0FBTztZQUNoQixZQUFZLEVBQUUsU0FBUyxDQUFDLFVBQVUsQ0FBQyxZQUFZO1lBQy9DLGFBQWEsRUFBRSxTQUFTLENBQUMsV0FBVyxDQUFDLGFBQWE7WUFDbEQsZ0JBQWdCLEVBQUcsU0FBUyxDQUFDLGdCQUFnQjtTQUM5QyxDQUFDLENBQUM7UUFFSCxTQUFTLENBQUMsY0FBYyxDQUFDLGNBQWMsQ0FBQyxhQUFhLEVBQUUsUUFBUSxDQUFDLE9BQU8sQ0FBQyxVQUFVLENBQUMsQ0FBQTtRQUNuRixTQUFTLENBQUMsY0FBYyxDQUFDLGNBQWMsQ0FBQyxtQkFBbUIsRUFBRSxRQUFRLENBQUMsZUFBZSxDQUFDLGlCQUFpQixDQUFDLENBQUE7UUFDeEcsU0FBUyxDQUFDLGNBQWMsQ0FBQyxjQUFjLENBQUMsb0JBQW9CLEVBQUUsUUFBUSxDQUFDLGdCQUFnQixDQUFDLGlCQUFpQixDQUFDLENBQUE7UUFDMUcsU0FBUyxDQUFDLGNBQWMsQ0FBQyxjQUFjLENBQUMsa0JBQWtCLEVBQUUsUUFBUSxDQUFDLGdCQUFnQixDQUFDLE1BQU0sQ0FBQyxDQUFBO1FBQzdGLFNBQVMsQ0FBQyxjQUFjLENBQUMsY0FBYyxDQUFDLGNBQWMsRUFBRSxRQUFRLENBQUMsZUFBZSxDQUFDLENBQUE7UUFDakYsU0FBUyxDQUFDLGNBQWMsQ0FBQyxjQUFjLENBQUMsZUFBZSxFQUFFLFFBQVEsQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFBO1FBQ25GLFNBQVMsQ0FBQyxjQUFjLENBQUMsY0FBYyxDQUFDLGNBQWMsRUFBRSxRQUFRLENBQUMsV0FBVyxDQUFDLGVBQWUsQ0FBQyxDQUFBO1FBQzdGLElBQUksV0FBVyxJQUFJLFNBQVMsS0FBSyxRQUFRLEVBQUU7WUFDekMsU0FBUyxDQUFDLGNBQWMsQ0FBQyxjQUFjLENBQUMsY0FBYyxFQUFFLFdBQVcsV0FBVyxDQUFDLGFBQWEsT0FBTyxDQUFDLENBQUE7U0FDckc7YUFBTSxJQUFJLFNBQVMsQ0FBQyxRQUFRLEVBQUU7WUFDN0IsU0FBUyxDQUFDLGNBQWMsQ0FBQyxjQUFjLENBQUMsY0FBYyxFQUFFLFNBQVMsQ0FBQyxRQUFRLENBQUMsbUJBQW1CLENBQUMsQ0FBQTtTQUNoRztRQUVEOztXQUVHO1FBQ0gsTUFBTSxpQkFBaUIsR0FBRyxJQUFJLG1DQUFjLENBQUMsU0FBUyxDQUFDLGNBQWMsQ0FBQyxDQUFDO1FBQ3ZFLE1BQU0sZUFBZSxHQUFHO1lBQ3RCLE1BQU0sRUFBRSxDQUFFLFNBQVMsQ0FBRTtZQUNyQixVQUFVLEVBQUUsQ0FBRSx1QkFBdUIsQ0FBRTtZQUN2QyxNQUFNLEVBQUUsRUFBRSxZQUFZLEVBQUUsQ0FBRSxRQUFRLENBQUMsT0FBTyxDQUFDLFVBQVUsQ0FBRSxFQUFFO1NBQzFELENBQUE7UUFDRCxJQUFJLGlCQUFJLENBQUMsSUFBSSxFQUFFLGlCQUFpQixFQUFFO1lBQ2hDLFlBQVksRUFBRSxlQUFlO1lBQzdCLE9BQU8sRUFBRSxDQUFDLGlCQUFpQixDQUFDO1NBQzdCLENBQUMsQ0FBQztRQUNILE1BQU0sdUJBQXVCLEdBQUc7WUFDOUIsTUFBTSxFQUFFLENBQUUsaUJBQWlCLENBQUU7WUFDN0IsVUFBVSxFQUFFO2dCQUNWLGdDQUFnQztnQkFDaEMsbUNBQW1DO2dCQUNuQyxrQ0FBa0M7Z0JBQ2xDLHFDQUFxQztnQkFDckMsc0NBQXNDO2dCQUN0Qyx5Q0FBeUM7YUFDMUM7U0FDRixDQUFBO1FBQ0QsSUFBSSxpQkFBSSxDQUFDLElBQUksRUFBRSx5QkFBeUIsRUFBRTtZQUN4QyxZQUFZLEVBQUUsdUJBQXVCO1lBQ3JDLE9BQU8sRUFBRSxDQUFDLGlCQUFpQixDQUFDO1NBQzdCLENBQUMsQ0FBQztRQUVILE1BQU0sTUFBTSxHQUFHLElBQUksd0JBQVUsQ0FBQyxJQUFJLEVBQUUsWUFBWSxFQUFFO1lBQ2hELEdBQUcsQ0FBQyxNQUFNLENBQUMsSUFBSSxFQUFDLE1BQU0sRUFBRSxNQUFNLENBQUMsYUFBYSxFQUFDO1lBQzdDLEdBQUcsQ0FBQyxXQUFXLENBQUMsSUFBSSxFQUFDLFdBQVcsRUFBRSxXQUFXLENBQUMsYUFBYSxFQUFDO1lBQzVELE1BQU0sRUFBRSxTQUFTLENBQUMsT0FBTyxDQUFDLEdBQUc7WUFDN0IsU0FBUyxFQUFFLFFBQVE7WUFDbkIsaUJBQWlCLEVBQUUsT0FBQSxTQUFTLENBQUMsUUFBUSwwQ0FBRSxVQUFVLEtBQUksRUFBRTtZQUN2RCw0QkFBNEIsRUFBRSxPQUFBLFNBQVMsQ0FBQyxpQkFBaUIsMENBQUUsZ0JBQWdCLEtBQUksRUFBRTtTQUNsRixDQUFDLENBQUM7UUFFSCxJQUFJLEdBQUcsQ0FBQyxTQUFTLENBQUMsSUFBSSxFQUFFLHNCQUFzQixFQUFFO1lBQzlDLEtBQUssRUFBRSxNQUFNLENBQUMsT0FBTyxDQUFDLGNBQWM7WUFDcEMsV0FBVyxFQUFFLHdDQUF3QztTQUN0RCxDQUFDLENBQUMsaUJBQWlCLENBQUMsZ0JBQWdCLENBQUMsQ0FBQztRQUN2QyxJQUFJLEdBQUcsQ0FBQyxTQUFTLENBQUMsSUFBSSxFQUFFLDBCQUEwQixFQUFFO1lBQ2xELEtBQUssRUFBRSxNQUFNLENBQUMsT0FBTyxDQUFDLHNCQUFzQjtZQUM1QyxXQUFXLEVBQUUseUVBQXlFO1NBQ3ZGLENBQUMsQ0FBQyxpQkFBaUIsQ0FBQyxrQkFBa0IsQ0FBQyxDQUFDO1FBQ3pDLElBQUksTUFBTSxFQUFFO1lBQ1YsSUFBSSxHQUFHLENBQUMsU0FBUyxDQUFDLElBQUksRUFBRSxjQUFjLEVBQUU7Z0JBQ3RDLEtBQUssRUFBRSxNQUFNLENBQUMsYUFBYTtnQkFDM0IsV0FBVyxFQUFFLDhCQUE4QjthQUM1QyxDQUFDLENBQUMsaUJBQWlCLENBQUMsUUFBUSxDQUFDLENBQUE7U0FDL0I7UUFDRCxJQUFJLFNBQVMsQ0FBQyxRQUFRLEVBQUU7WUFDdEIsSUFBSSxHQUFHLENBQUMsU0FBUyxDQUFDLElBQUksRUFBRSxrQkFBa0IsRUFBRTtnQkFDMUMsS0FBSyxFQUFFLFNBQVMsQ0FBQyxRQUFRLENBQUMsVUFBVTtnQkFDcEMsV0FBVyxFQUFFLHVEQUF1RDthQUNyRSxDQUFDLENBQUMsaUJBQWlCLENBQUMsWUFBWSxDQUFDLENBQUE7U0FDbkM7SUFFSCxDQUFDO0lBMU1TLFNBQVMsQ0FBQyxFQUFVLEVBQUUsS0FBNkIsSUFBc0IsT0FBTyxJQUFJLEdBQUcsQ0FBQyxZQUFZLENBQUMsSUFBSSxFQUFFLEVBQUUsRUFBRSxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDeEgsYUFBYSxDQUFDLEtBQTJDOztRQUNqRSxLQUFLLE1BQU0sR0FBRyxJQUFJLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLEVBQUU7WUFDcEMsTUFBTSxNQUFNLEdBQUcsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1lBQzFCLElBQUksQ0FBQyxXQUFXLENBQUMsR0FBRyxDQUFDLEdBQUcsTUFBTSxDQUFDLE1BQU0sT0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLEdBQUcsQ0FBQyxtQ0FBSSxFQUFFLENBQUMsQ0FBQztTQUNwRTtRQUNELElBQUksQ0FBQyxlQUFlLEVBQUUsQ0FBQztJQUN6QixDQUFDO0lBQ08sZUFBZTtRQUNyQixJQUFJLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxRQUFRLEVBQUU7WUFBRSxJQUFJLENBQUMsZUFBZSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUM7U0FBRTtRQUMzRSxNQUFNLEtBQUssR0FBRyxDQUFDLEtBQWEsRUFBRSxNQUEwQixFQUFFLEVBQUU7WUFDMUQsT0FBTztnQkFDTCxLQUFLLEVBQUUsRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFFO2dCQUN6QixVQUFVLEVBQUUsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsRUFBRTtvQkFDekIsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztnQkFDOUIsQ0FBQyxDQUFDLENBQUMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDO2FBQ3BCLENBQUM7UUFDSixDQUFDLENBQUM7UUFDRixJQUFJLENBQUMsZUFBZSxDQUFDLFFBQVEsQ0FBQyxnQ0FBZ0MsQ0FBQyxHQUFHO1lBQ2hFLGVBQWUsRUFBRSxNQUFNLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsR0FBRyxFQUFFLElBQUksQ0FBQyxXQUFXLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBRTtTQUM5RixDQUFDO0lBQ0osQ0FBQztDQXNMRjtBQUVELE1BQU0sR0FBRyxHQUFHLElBQUksR0FBRyxDQUFDLEdBQUcsRUFBRSxDQUFDO0FBQzFCLElBQUksVUFBVSxDQUFDLEdBQUcsRUFBRSxRQUFRLENBQUMsQ0FBQztBQUU5QixHQUFHLENBQUMsS0FBSyxFQUFFLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgeyBTdGF0aWNTaXRlIH0gZnJvbSAnLi9zdGF0aWMtc2l0ZSc7XG5pbXBvcnQgeyBMYW1iZGFBcGlTdGFjayB9IGZyb20gXCIuL2xhbWJkYS1hcGktc3RhY2tcIlxuaW1wb3J0IHsgRWNzU3RhY2sgfSBmcm9tIFwiLi9lY3Mtc3RhY2tcIlxuaW1wb3J0IHsgUnVsZSB9IGZyb20gJ0Bhd3MtY2RrL2F3cy1ldmVudHMnO1xuaW1wb3J0IHsgTGFtYmRhRnVuY3Rpb24gfSBmcm9tICdAYXdzLWNkay9hd3MtZXZlbnRzLXRhcmdldHMnO1xuaW1wb3J0ICogYXMgZWMyIGZyb20gJ0Bhd3MtY2RrL2F3cy1lYzInO1xuaW1wb3J0ICogYXMgY2RrIGZyb20gXCJAYXdzLWNkay9jb3JlXCI7XG5cbmV4cG9ydCBjb25zdCBlbnVtIEF1dGhUeXBlIHtcbiAgQ09HTklUTyA9IFwiY29nbml0b1wiLFxuICBPUEVOSUQgPSBcIm9wZW5pZFwiXG59XG5cbmludGVyZmFjZSBNTEJvdFNldHRpbmdzIHtcbiAgdnBjPzogZWMyLklWcGM7XG4gIC8vIHB1YmxpY1N1Ym5ldHM6IGVjMi5TdWJuZXRTZWxlY3Rpb247XG4gIHByaXZhdGVTdWJuZXRzPzogZWMyLlN1Ym5ldFNlbGVjdGlvbjtcbn1cblxuY2xhc3MgTUxCb3RTdGFjayBleHRlbmRzIGNkay5TdGFjayB7XG4gIHByaXZhdGUgX3BhcmFtR3JvdXA6IHsgW2dycG5hbWU6IHN0cmluZ106IGNkay5DZm5QYXJhbWV0ZXJbXX0gPSB7fVxuICBwcml2YXRlIF9tbGJvdFNldHRpbmdzOiBNTEJvdFNldHRpbmdzID0geyB9O1xuXG4gIHByb3RlY3RlZCBtYWtlUGFyYW0oaWQ6IHN0cmluZywgcHJvcHM/OiBjZGsuQ2ZuUGFyYW1ldGVyUHJvcHMpOiBjZGsuQ2ZuUGFyYW1ldGVyIHsgcmV0dXJuIG5ldyBjZGsuQ2ZuUGFyYW1ldGVyKHRoaXMsIGlkLCBwcm9wcyk7IH1cbiAgcHJvdGVjdGVkIGFkZEdyb3VwUGFyYW0ocHJvcHM6IHsgW2tleTogc3RyaW5nXTogY2RrLkNmblBhcmFtZXRlcltdfSk6IHZvaWQge1xuICAgIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHByb3BzKSkge1xuICAgICAgY29uc3QgcGFyYW1zID0gcHJvcHNba2V5XTtcbiAgICAgIHRoaXMuX3BhcmFtR3JvdXBba2V5XSA9IHBhcmFtcy5jb25jYXQodGhpcy5fcGFyYW1Hcm91cFtrZXldID8/IFtdKTtcbiAgICB9XG4gICAgdGhpcy5fc2V0UGFyYW1Hcm91cHMoKTtcbiAgfVxuICBwcml2YXRlIF9zZXRQYXJhbUdyb3VwcygpOiB2b2lkIHtcbiAgICBpZiAoIXRoaXMudGVtcGxhdGVPcHRpb25zLm1ldGFkYXRhKSB7IHRoaXMudGVtcGxhdGVPcHRpb25zLm1ldGFkYXRhID0ge307IH1cbiAgICBjb25zdCBta2dycCA9IChsYWJlbDogc3RyaW5nLCBwYXJhbXM6IGNkay5DZm5QYXJhbWV0ZXJbXSkgPT4ge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgTGFiZWw6IHsgZGVmYXVsdDogbGFiZWwgfSxcbiAgICAgICAgUGFyYW1ldGVyczogcGFyYW1zLm1hcChwID0+IHtcbiAgICAgICAgICByZXR1cm4gcCA/IHAubG9naWNhbElkIDogJyc7XG4gICAgICAgIH0pLmZpbHRlcihpZCA9PiBpZCksXG4gICAgICB9O1xuICAgIH07XG4gICAgdGhpcy50ZW1wbGF0ZU9wdGlvbnMubWV0YWRhdGFbJ0FXUzo6Q2xvdWRGb3JtYXRpb246OkludGVyZmFjZSddID0ge1xuICAgICAgUGFyYW1ldGVyR3JvdXBzOiBPYmplY3Qua2V5cyh0aGlzLl9wYXJhbUdyb3VwKS5tYXAoa2V5ID0+IG1rZ3JwKGtleSwgdGhpcy5fcGFyYW1Hcm91cFtrZXldKSApLFxuICAgIH07XG4gIH1cblxuICBjb25zdHJ1Y3RvcihwYXJlbnQ6IGNkay5BcHAsIG5hbWU6IHN0cmluZywgcHJvcHM/OiBjZGsuU3RhY2tQcm9wcykge1xuICAgIHN1cGVyKHBhcmVudCwgbmFtZSwgcHJvcHMpO1xuXG4gICAgY29uc3QgdmVyc2lvbiA9IChwcm9jZXNzLmVudi5WRVJTSU9OICYmIHByb2Nlc3MuZW52LlZFUlNJT04uc3RhcnRzV2l0aCgndicpKSA/IHByb2Nlc3MuZW52LlZFUlNJT04gOiAndjEuMS4yJ1xuICAgIFxuICAgIGNvbnN0IHBhcnRpdGlvbiA9IHBhcmVudC5ub2RlLnRyeUdldENvbnRleHQoJ1BhcnRpdGlvbicpO1xuICAgIGNvbnN0IGV4aXN0aW5nVnBjID0gcGFyZW50Lm5vZGUudHJ5R2V0Q29udGV4dCgnRXhpc3RpbmdWUEMnKTtcbiAgICBjb25zdCB2cGNNc2cgPSBleGlzdGluZ1ZwYyA/ICdleGlzdGluZyB2cGMnIDogJ25ldyB2cGMnO1xuICAgIHRoaXMudGVtcGxhdGVPcHRpb25zLmRlc2NyaXB0aW9uID0gYChTTzgwMTIpIE1hY2hpbmUgTGVhcm5pbmcgQm90IHdpdGggJHt2cGNNc2d9LiBUZW1wbGF0ZSB2ZXJzaW9uICR7dmVyc2lvbn1gXG4gICAgbGV0IGF1dGhUeXBlO1xuICAgIGxldCBkb21haW4sIGxvZ2luRG9tYWluOyAvLyBzc2xDZXJ0aWZpY2F0ZUlkLFxuICAgIGxldCB2cGM7XG5cbiAgICBpZiAocGFydGl0aW9uID09PSAnYXdzLWNuJykge1xuICAgICAgYXV0aFR5cGUgPSBBdXRoVHlwZS5PUEVOSUQ7XG4gICAgfSBlbHNlIHtcbiAgICAgIGF1dGhUeXBlID0gQXV0aFR5cGUuQ09HTklUTztcbiAgICB9XG5cbiAgICBpZiAocGFydGl0aW9uID09PSAnYXdzLWNuJykge1xuICAgICAgZG9tYWluID0gbmV3IGNkay5DZm5QYXJhbWV0ZXIodGhpcywgJ0RvbWFpbicsIHtcbiAgICAgICAgZGVzY3JpcHRpb246ICdEb21haW4gTmFtZScsXG4gICAgICAgIHR5cGU6ICdTdHJpbmcnLFxuICAgICAgICBhbGxvd2VkUGF0dGVybjogJyg/OlthLXowLTldKD86W2EtejAtOS1dezAsNjF9W2EtejAtOV0pP1xcXFwuKStbYS16MC05XVthLXowLTktXXswLDYxfVthLXpdJ1xuICAgICAgfSk7XG5cbiAgICAgIGxvZ2luRG9tYWluID0gbmV3IGNkay5DZm5QYXJhbWV0ZXIodGhpcywgJ0xvZ2luRG9tYWluJywge1xuICAgICAgICB0eXBlOiAnU3RyaW5nJyxcbiAgICAgICAgYWxsb3dlZFBhdHRlcm46ICcoPzpbYS16MC05XSg/OlthLXowLTktXXswLDYxfVthLXowLTldKT9cXFxcLikrW2EtejAtOV1bYS16MC05LV17MCw2MX1bYS16XScsXG4gICAgICAgIGRlc2NyaXB0aW9uOiAnVGhlIGRvbWFpbiBhZGRyZXNzIGZvciBPSURDIGJhc2VkIGxvZ2luLCBlLmcuIG1sLWJvdC5hdXRoaW5nLmNuJ1xuICAgICAgfSlcblxuICAgICAgdGhpcy50ZW1wbGF0ZU9wdGlvbnMubWV0YWRhdGEgPSB7XG4gICAgICAgICdBV1M6OkNsb3VkRm9ybWF0aW9uOjpJbnRlcmZhY2UnOiB7XG4gICAgICAgICAgUGFyYW1ldGVyR3JvdXBzOiBbXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgIExhYmVsOiB7IGRlZmF1bHQ6ICdQb3J0YWwnIH0sXG4gICAgICAgICAgICAgIFBhcmFtZXRlcnM6IFsgZG9tYWluLmxvZ2ljYWxJZCwgbG9naW5Eb21haW4ubG9naWNhbElkIF0gLy8gc3NsQ2VydGlmaWNhdGVJZC5sb2dpY2FsSWQsIFxuICAgICAgICAgICAgfVxuICAgICAgICAgIF0sXG4gICAgICAgICAgUGFyYW1ldGVyTGFiZWxzOiB7XG4gICAgICAgICAgICAvLyBTU0xDZXJ0aWZpY2F0ZUlkOiB7IGRlZmF1bHQ6ICdTU0wgQ2VydGlmaWNhdGUgSUQnIH0sXG4gICAgICAgICAgICBEb21haW46IHsgZGVmYXVsdDogJ0RvbWFpbicgfSxcbiAgICAgICAgICAgIExvZ2luRG9tYWluOiB7IGRlZmF1bHQ6ICdMb2dpbiBEb21haW4nIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICBpZiAoZXhpc3RpbmdWcGMgPT0gXCJ0cnVlXCIpIHtcbiAgICAgIGNvbnN0IHZwY0lkUGFyYW0gPSB0aGlzLm1ha2VQYXJhbSgnVnBjSWQnLCB7XG4gICAgICAgIHR5cGU6ICdBV1M6OkVDMjo6VlBDOjpJZCcsXG4gICAgICAgIGRlc2NyaXB0aW9uOiAnWW91ciBWUEMgSWQnLFxuICAgICAgfSk7XG4gICAgICAvLyBjb25zdCBwdWJTdWJuZXRzUGFyYW0gPSB0aGlzLm1ha2VQYXJhbSgnUHViU3VibmV0cycsIHtcbiAgICAgIC8vICAgdHlwZTogJ0xpc3Q8QVdTOjpFQzI6OlN1Ym5ldDo6SWQ+JyxcbiAgICAgIC8vICAgZGVzY3JpcHRpb246ICdQdWJsaWMgc3VibmV0cyAoQ2hvb3NlIHR3byknLFxuICAgICAgLy8gfSk7XG4gICAgICBjb25zdCBwcml2U3VibmV0c1BhcmFtID0gdGhpcy5tYWtlUGFyYW0oJ1ByaXZTdWJuZXRzJywge1xuICAgICAgICB0eXBlOiAnTGlzdDxBV1M6OkVDMjo6U3VibmV0OjpJZD4nLFxuICAgICAgICBkZXNjcmlwdGlvbjogJ1ByaXZhdGUgc3VibmV0cyAoQ2hvb3NlIHR3byknLFxuICAgICAgfSk7XG4gICAgICB0aGlzLmFkZEdyb3VwUGFyYW0oeyAnVlBDIFNldHRpbmdzJzogW3ZwY0lkUGFyYW0sIHByaXZTdWJuZXRzUGFyYW1dIH0pOyAvLyBwdWJTdWJuZXRzUGFyYW0sIFxuXG4gICAgICBjb25zdCBhenMgPSBbJ2EnLCAnYiddO1xuICAgICAgdnBjID0gZWMyLlZwYy5mcm9tVnBjQXR0cmlidXRlcyh0aGlzLCAnVnBjQXR0cicsIHtcbiAgICAgICAgdnBjSWQ6IHZwY0lkUGFyYW0udmFsdWVBc1N0cmluZyxcbiAgICAgICAgdnBjQ2lkckJsb2NrOiBjZGsuQXdzLk5PX1ZBTFVFLFxuICAgICAgICBhdmFpbGFiaWxpdHlab25lczogYXpzLFxuICAgICAgICAvLyBwdWJsaWNTdWJuZXRJZHM6IGF6cy5tYXAoKF8sIGluZGV4KSA9PiBGbi5zZWxlY3QoaW5kZXgsIHB1YlN1Ym5ldHNQYXJhbS52YWx1ZUFzTGlzdCkpLFxuICAgICAgICBwcml2YXRlU3VibmV0SWRzOiBhenMubWFwKChfLCBpbmRleCkgPT4gY2RrLkZuLnNlbGVjdChpbmRleCwgcHJpdlN1Ym5ldHNQYXJhbS52YWx1ZUFzTGlzdCkpLFxuICAgICAgfSk7XG5cbiAgICAgIE9iamVjdC5hc3NpZ24odGhpcy5fbWxib3RTZXR0aW5ncywge1xuICAgICAgICB2cGMsXG4gICAgICAgIC8vIHB1YmxpY1N1Ym5ldHM6IHsgc3VibmV0czogdnBjLnB1YmxpY1N1Ym5ldHMgfSxcbiAgICAgICAgcHJpdmF0ZVN1Ym5ldHM6IHsgc3VibmV0czogdnBjLnByaXZhdGVTdWJuZXRzIH0sXG4gICAgICB9KTtcbiAgICB9IGVsc2Uge1xuICAgICAgdnBjID0gbmV3IGVjMi5WcGModGhpcywgJ1ZwYycsIHsgbWF4QXpzOiAyLCBuYXRHYXRld2F5czogMSB9KTtcbiAgICAgIE9iamVjdC5hc3NpZ24odGhpcy5fbWxib3RTZXR0aW5ncywge1xuICAgICAgICB2cGMsXG4gICAgICAgIC8vIHB1YmxpY1N1Ym5ldHM6IHsgc3VibmV0czogdnBjLnB1YmxpY1N1Ym5ldHMgfSxcbiAgICAgICAgcHJpdmF0ZVN1Ym5ldHM6IHsgc3VibmV0czogdnBjLnByaXZhdGVTdWJuZXRzIH0sXG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBjb25zdCBwcml2YXRlU3VibmV0cyA9IHZwYy5wcml2YXRlU3VibmV0c1syXSA/IFxuICAgICAgdnBjLnByaXZhdGVTdWJuZXRzWzBdLnN1Ym5ldElkICsgXCIsXCIgKyB2cGMucHJpdmF0ZVN1Ym5ldHNbMV0uc3VibmV0SWQgKyBcIixcIiArIHZwYy5wcml2YXRlU3VibmV0c1syXS5zdWJuZXRJZCA6XG4gICAgICB2cGMucHJpdmF0ZVN1Ym5ldHNbMF0uc3VibmV0SWQgKyBcIixcIiArIHZwYy5wcml2YXRlU3VibmV0c1sxXS5zdWJuZXRJZDtcblxuICAgIGNvbnN0IGxhbWJkYUFwaSA9IG5ldyBMYW1iZGFBcGlTdGFjayh0aGlzLCAnTGFtYmRhQXBpJywge1xuICAgICAgYXV0aFR5cGU6IGF1dGhUeXBlLFxuICAgICAgdnBjOiB2cGMsXG4gICAgICBwcml2YXRlU3VibmV0czogcHJpdmF0ZVN1Ym5ldHMsXG4gICAgfSk7XG5cbiAgICBjb25zdCBlY3NTdGFjayA9IG5ldyBFY3NTdGFjayh0aGlzLCAnRWNzU3RhY2snLCB7XG4gICAgICBkYXRhQnVja2V0OiBsYW1iZGFBcGkuZGF0YUJ1Y2tldCxcbiAgICAgIG1vZGVsQnVja2V0OiBsYW1iZGFBcGkubW9kZWxCdWNrZXQsXG4gICAgICB2cGM6IHZwYyxcbiAgICAgIHZlcnNpb246IHZlcnNpb24sXG4gICAgICBmaWxlU3lzdGVtSWQ6IGxhbWJkYUFwaS5maWxlU3lzdGVtLmZpbGVTeXN0ZW1JZCxcbiAgICAgIGFjY2Vzc1BvaW50SWQ6IGxhbWJkYUFwaS5hY2Nlc3NQb2ludC5hY2Nlc3NQb2ludElkLFxuICAgICAgZWMyU2VjdXJpdHlHcm91cCA6IGxhbWJkYUFwaS5lYzJTZWN1cml0eUdyb3VwLFxuICAgIH0pO1xuXG4gICAgbGFtYmRhQXBpLmxhbWJkYUZ1bmN0aW9uLmFkZEVudmlyb25tZW50KFwiQ0xVU1RFUl9BUk5cIiwgZWNzU3RhY2suY2x1c3Rlci5jbHVzdGVyQXJuKVxuICAgIGxhbWJkYUFwaS5sYW1iZGFGdW5jdGlvbi5hZGRFbnZpcm9ubWVudChcIlRSQUlOSU5HX1RBU0tfQVJOXCIsIGVjc1N0YWNrLnRyYWluaW5nVGFza0RlZi50YXNrRGVmaW5pdGlvbkFybilcbiAgICBsYW1iZGFBcGkubGFtYmRhRnVuY3Rpb24uYWRkRW52aXJvbm1lbnQoXCJJTkZFUkVOQ0VfVEFTS19BUk5cIiwgZWNzU3RhY2suaW5mZXJlbmNlVGFza0RlZi50YXNrRGVmaW5pdGlvbkFybilcbiAgICBsYW1iZGFBcGkubGFtYmRhRnVuY3Rpb24uYWRkRW52aXJvbm1lbnQoXCJJTkZFUkVOQ0VfRkFNSUxZXCIsIGVjc1N0YWNrLmluZmVyZW5jZVRhc2tEZWYuZmFtaWx5KVxuICAgIGxhbWJkYUFwaS5sYW1iZGFGdW5jdGlvbi5hZGRFbnZpcm9ubWVudChcIlRSQUlOSU5HX0FTR1wiLCBlY3NTdGFjay50cmFpbmluZ0FzZ05hbWUpXG4gICAgbGFtYmRhQXBpLmxhbWJkYUZ1bmN0aW9uLmFkZEVudmlyb25tZW50KFwiSU5GRVJFTkNFX0FTR1wiLCBlY3NTdGFjay5pbmZlcmVuY2VBc2dOYW1lKVxuICAgIGxhbWJkYUFwaS5sYW1iZGFGdW5jdGlvbi5hZGRFbnZpcm9ubWVudChcIklORkVSRU5DRV9TR1wiLCBlY3NTdGFjay5pbmZlcmVuY2VTRy5zZWN1cml0eUdyb3VwSWQpXG4gICAgaWYgKGxvZ2luRG9tYWluICYmIHBhcnRpdGlvbiA9PT0gJ2F3cy1jbicpIHtcbiAgICAgIGxhbWJkYUFwaS5sYW1iZGFGdW5jdGlvbi5hZGRFbnZpcm9ubWVudChcIkxPR0lOX0RPTUFJTlwiLCBgaHR0cHM6Ly8ke2xvZ2luRG9tYWluLnZhbHVlQXNTdHJpbmd9L29pZGNgKVxuICAgIH0gZWxzZSBpZiAobGFtYmRhQXBpLnVzZXJQb29sKSB7XG4gICAgICBsYW1iZGFBcGkubGFtYmRhRnVuY3Rpb24uYWRkRW52aXJvbm1lbnQoXCJMT0dJTl9ET01BSU5cIiwgbGFtYmRhQXBpLnVzZXJQb29sLnVzZXJQb29sUHJvdmlkZXJVcmwpXG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogQ3JlYXRlIGEgRXZlbnRCcmlkZ2UgcnVsZSB0byBhdXRvbWF0ZSBtb2RlbCBkZXBsb3ltZW50IHByb2Nlc3MgdXNpbmcgTGFtYmRhXG4gICAgICovXG4gICAgY29uc3QgZXZlbnRCcmlkZ2VUYXJnZXQgPSBuZXcgTGFtYmRhRnVuY3Rpb24obGFtYmRhQXBpLmxhbWJkYUZ1bmN0aW9uKTtcbiAgICBjb25zdCBlY3NFdmVudFBhdHRlcm4gPSB7XG4gICAgICBzb3VyY2U6IFsgXCJhd3MuZWNzXCIgXSxcbiAgICAgIGRldGFpbFR5cGU6IFsgXCJFQ1MgVGFzayBTdGF0ZSBDaGFuZ2VcIiBdLFxuICAgICAgZGV0YWlsOiB7IFwiY2x1c3RlckFyblwiOiBbIGVjc1N0YWNrLmNsdXN0ZXIuY2x1c3RlckFybiBdIH1cbiAgICB9XG4gICAgbmV3IFJ1bGUodGhpcywgJ0Vjc0V2ZW50UGF0dGVybicsIHtcbiAgICAgIGV2ZW50UGF0dGVybjogZWNzRXZlbnRQYXR0ZXJuLFxuICAgICAgdGFyZ2V0czogW2V2ZW50QnJpZGdlVGFyZ2V0XSxcbiAgICB9KTtcbiAgICBjb25zdCBhdXRvc2NhbGluZ0V2ZW50UGF0dGVybiA9IHtcbiAgICAgIHNvdXJjZTogWyBcImF3cy5hdXRvc2NhbGluZ1wiIF0sXG4gICAgICBkZXRhaWxUeXBlOiBbXG4gICAgICAgIFwiRUMyIEluc3RhbmNlIExhdW5jaCBTdWNjZXNzZnVsXCIsXG4gICAgICAgIFwiRUMyIEluc3RhbmNlIFRlcm1pbmF0ZSBTdWNjZXNzZnVsXCIsXG4gICAgICAgIFwiRUMyIEluc3RhbmNlIExhdW5jaCBVbnN1Y2Nlc3NmdWxcIixcbiAgICAgICAgXCJFQzIgSW5zdGFuY2UgVGVybWluYXRlIFVuc3VjY2Vzc2Z1bFwiLFxuICAgICAgICBcIkVDMiBJbnN0YW5jZS1sYXVuY2ggTGlmZWN5Y2xlIEFjdGlvblwiLFxuICAgICAgICBcIkVDMiBJbnN0YW5jZS10ZXJtaW5hdGUgTGlmZWN5Y2xlIEFjdGlvblwiXG4gICAgICBdLFxuICAgIH1cbiAgICBuZXcgUnVsZSh0aGlzLCAnQXV0b3NjYWxpbmdFdmVudFBhdHRlcm4nLCB7XG4gICAgICBldmVudFBhdHRlcm46IGF1dG9zY2FsaW5nRXZlbnRQYXR0ZXJuLFxuICAgICAgdGFyZ2V0czogW2V2ZW50QnJpZGdlVGFyZ2V0XSxcbiAgICB9KTtcbiAgICBcbiAgICBjb25zdCBwb3J0YWwgPSBuZXcgU3RhdGljU2l0ZSh0aGlzLCAnU3RhdGljU2l0ZScsIHtcbiAgICAgIC4uLihkb21haW4pICYmIHtkb21haW46IGRvbWFpbi52YWx1ZUFzU3RyaW5nfSxcbiAgICAgIC4uLihsb2dpbkRvbWFpbikgJiYge2xvZ2luRG9tYWluOiBsb2dpbkRvbWFpbi52YWx1ZUFzU3RyaW5nfSxcbiAgICAgIGFwaVVybDogbGFtYmRhQXBpLnJlc3RBcGkudXJsLFxuICAgICAgYXV0aF90eXBlOiBhdXRoVHlwZSxcbiAgICAgIGF3c191c2VyX3Bvb2xzX2lkOiBsYW1iZGFBcGkudXNlclBvb2w/LnVzZXJQb29sSWQgfHwgJycsXG4gICAgICBhd3NfdXNlcl9wb29sc193ZWJfY2xpZW50X2lkOiBsYW1iZGFBcGkudXNlclBvb2xBcGlDbGllbnQ/LnVzZXJQb29sQ2xpZW50SWQgfHwgJycsXG4gICAgfSk7XG5cbiAgICBuZXcgY2RrLkNmbk91dHB1dCh0aGlzLCAnRGlzdHJpYnV0aW9uSWRPdXRwdXQnLCB7XG4gICAgICB2YWx1ZTogcG9ydGFsLndlYnNpdGUuZGlzdHJpYnV0aW9uSWQsXG4gICAgICBkZXNjcmlwdGlvbjogJ1RoZSBQb3J0YWwgQ2xvdWRGcm9udCBEaXN0cmlidXRpb24gSUQuJ1xuICAgIH0pLm92ZXJyaWRlTG9naWNhbElkKCdEaXN0cmlidXRpb25JZCcpO1xuICAgIG5ldyBjZGsuQ2ZuT3V0cHV0KHRoaXMsICdEaXN0cmlidXRpb25Eb21haW5PdXRwdXQnLCB7XG4gICAgICB2YWx1ZTogcG9ydGFsLndlYnNpdGUuZGlzdHJpYnV0aW9uRG9tYWluTmFtZSxcbiAgICAgIGRlc2NyaXB0aW9uOiAnVGhlIGRvbWFpbiBuYW1lIG9mIHRoZSBwb3J0YWwuIENyZWF0ZSBhIENOQU1FIGFuZCBwb2ludCB0byB0aGlzIHJlY29yZC4nXG4gICAgfSkub3ZlcnJpZGVMb2dpY2FsSWQoJ0Nsb3VkRnJvbnREb21haW4nKTtcbiAgICBpZiAoZG9tYWluKSB7XG4gICAgICBuZXcgY2RrLkNmbk91dHB1dCh0aGlzLCAnRG9tYWluT3V0cHV0Jywge1xuICAgICAgICB2YWx1ZTogZG9tYWluLnZhbHVlQXNTdHJpbmcsXG4gICAgICAgIGRlc2NyaXB0aW9uOiAnVGhlIGRvbWFpbiBVUkwgb2YgdGhlIHBvcnRhbCdcbiAgICAgIH0pLm92ZXJyaWRlTG9naWNhbElkKCdEb21haW4nKVxuICAgIH1cbiAgICBpZiAobGFtYmRhQXBpLnVzZXJQb29sKSB7XG4gICAgICBuZXcgY2RrLkNmbk91dHB1dCh0aGlzLCAnVXNlclBvb2xJZE91dHB1dCcsIHtcbiAgICAgICAgdmFsdWU6IGxhbWJkYUFwaS51c2VyUG9vbC51c2VyUG9vbElkLFxuICAgICAgICBkZXNjcmlwdGlvbjogJ1RoZSBDb2duaXRvIFVzZXJQb29sIGZvciBtYW5hZ2luZyB1c2VycyBvZiB0aGUgcG9ydGFsJ1xuICAgICAgfSkub3ZlcnJpZGVMb2dpY2FsSWQoJ1VzZXJQb29sSWQnKVxuICAgIH1cblxuICB9XG59XG5cbmNvbnN0IGFwcCA9IG5ldyBjZGsuQXBwKCk7XG5uZXcgTUxCb3RTdGFjayhhcHAsICdtbC1ib3QnKTtcblxuYXBwLnN5bnRoKCk7XG4iXX0=