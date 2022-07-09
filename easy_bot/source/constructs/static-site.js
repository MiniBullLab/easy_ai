"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.StaticSite = void 0;
const cloudfront = require("@aws-cdk/aws-cloudfront");
const aws_cloudfront_1 = require("@aws-cdk/aws-cloudfront");
const origins = require("@aws-cdk/aws-cloudfront-origins");
const s3 = require("@aws-cdk/aws-s3");
const s3deploy = require("@aws-cdk/aws-s3-deployment");
const cdk = require("@aws-cdk/core");
const core_1 = require("@aws-cdk/core");
const iam = require("@aws-cdk/aws-iam");
const lambda = require("@aws-cdk/aws-lambda");
const path = require("path");
/**
 * Static site infrastructure, which deploys site content to an S3 bucket.
 *
 * The site redirects from HTTP to HTTPS, using a CloudFront distribution,
 * Route53 alias record, and ACM certificate.
 */
class StaticSite extends core_1.Construct {
    constructor(parent, name, props) {
        super(parent, name);
        const partition = parent.node.tryGetContext('Partition');
        const websiteBucket = new s3.Bucket(this, 'WebsiteBucket', {
            versioned: false,
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: true
        });
        // @ts-ignore
        this.website = new cloudfront.Distribution(this, 'WebsiteDistribution', {
            defaultBehavior: {
                origin: new origins.S3Origin(websiteBucket),
                viewerProtocolPolicy: aws_cloudfront_1.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            },
            defaultRootObject: 'index.html',
            ...(partition === 'aws-cn') && {
                domainNames: [props.domain],
                enableIpv6: false,
                priceClass: cloudfront.PriceClass.PRICE_CLASS_ALL
            }
        });
        if (partition === 'aws-cn') {
            /**
             *  Generated Default Cache Behavior
             *  "DefaultCacheBehavior": {
             *       "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
             *       "Compress": true,
             *       "TargetOriginId": "mlbotStaticSiteCloudFrontToS3CloudFrontDistributionOrigin17913D4C5",
             *       "ViewerProtocolPolicy": "redirect-to-https"
             *     },
             *
             *   CachePolicyId is not supported in aws-cn partition
             *   Expected Default Cache Behavior
             *
             *   "DefaultCacheBehavior": {
             *       "AllowedMethods": [ "GET", "HEAD" ],
             *       "TargetOriginId": "mlbotStaticSiteCloudFrontToS3CloudFrontDistributionOrigin17913D4C5",
             *       "ForwardedValues": {
             *           "QueryString": false,
             *           "Headers": [ "Origin", "Accept" ],
             *           "Cookies": { "Forward": "none" }
             *       },
             *       "ViewerProtocolPolicy": "redirect-to-https"
             *   },
             *
             */
            const cfnDistribution = this.website.node.defaultChild;
            cfnDistribution.addPropertyDeletionOverride('DistributionConfig.DefaultCacheBehavior.CachePolicyId');
            cfnDistribution.addPropertyOverride('DistributionConfig.DefaultCacheBehavior.AllowedMethods', ["GET", "HEAD"]);
            cfnDistribution.addPropertyOverride('DistributionConfig.DefaultCacheBehavior.ForwardedValues', {
                QueryString: false,
                Headers: ["Origin", "Accept"],
                Cookies: { Forward: "none" }
            });
            /**
             * cloudfront.Distribution does not allow SSL Certificate in IAM.
             */
            // cfnDistribution.addPropertyOverride('DistributionConfig.ViewerCertificate', {
            //   IamCertificateId: {
            //     Ref: 'SSLCertificateId'
            //   },
            //   SslSupportMethod: 'sni-only',
            //   MinimumProtocolVersion: 'TLSv1.2_2018'
            // })
        }
        //-------------------------------------------------------
        // Custom Resources
        //-------------------------------------------------------
        // CustomResourceRole
        const customResourceRole = new iam.Role(this, 'CustomResourceRole', {
            assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
            path: '/',
            roleName: `${cdk.Aws.STACK_NAME}CustomResourceRole-${cdk.Aws.REGION}`
        });
        const cfnCustomResourceRole = customResourceRole.node.defaultChild;
        cfnCustomResourceRole.overrideLogicalId('CustomResourceRole');
        // CustomResourcePolicy
        const customResourcePolicy = new iam.Policy(this, 'CustomResourcePolicy', {
            policyName: `${cdk.Aws.STACK_NAME}CustomResourcePolicy`,
            statements: [
                new iam.PolicyStatement({
                    actions: [
                        'logs:CreateLogStream',
                        'logs:CreateLogGroup',
                        'logs:PutLogEvents'
                    ],
                    resources: [
                        `arn:${cdk.Aws.PARTITION}:logs:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:log-group:/aws/lambda/*`
                    ]
                }),
                new iam.PolicyStatement({
                    actions: ['s3:GetObject', 's3:PutObject', 's3:ListBucket'],
                    resources: [`arn:${cdk.Aws.PARTITION}:s3:::*`]
                })
            ]
        });
        customResourcePolicy.attachToRole(customResourceRole);
        const cfnCustomResourcePolicy = customResourcePolicy.node.defaultChild;
        cfnCustomResourcePolicy.overrideLogicalId('CustomResourcePolicy');
        const customResourceFunction = new lambda.Function(this, 'CustomHandler', {
            description: 'AWS ML Bot - Custom resource',
            runtime: lambda.Runtime.NODEJS_12_X,
            handler: 'index.handler',
            timeout: cdk.Duration.seconds(30),
            memorySize: 512,
            role: customResourceRole,
            code: lambda.Code.fromAsset(path.join(__dirname, '../custom-resource/'), {
                bundling: {
                    image: lambda.Runtime.NODEJS_12_X.bundlingDockerImage,
                    command: [
                        'bash', '-c', [
                            `cd /asset-output/`,
                            `cp -r /asset-input/* /asset-output/`,
                            `cd /asset-output/`,
                            `npm install`
                        ].join(' && ')
                    ],
                    user: 'root'
                }
            })
        });
        // CustomResourceConfig
        this.createCustomResource('CustomResourceConfig', customResourceFunction, {
            properties: [
                { path: 'Region', value: cdk.Aws.REGION },
                { path: 'destS3Bucket', value: websiteBucket.bucketName },
                { path: 'destS3key', value: 'aws-exports.json' },
                { path: 'customAction', value: 'putConfigFile' },
                {
                    path: 'configItem', value: {
                        apiUrl: props.apiUrl,
                        aws_project_region: cdk.Aws.REGION,
                        authType: props.auth_type === "openid" /* OPENID */ ? 'OPENID' : 'COGNITO',
                        aws_user_pools_id: props.aws_user_pools_id,
                        aws_user_pools_web_client_id: props.aws_user_pools_web_client_id,
                        ...(partition === 'aws-cn') && {
                            aws_oidc_token_validation_url: `https://${props.loginDomain}/api/v2/oidc/validate_token`,
                            aws_oidc_provider: `https://${props.loginDomain}/oidc`,
                            aws_oidc_logout_url: `https://${props.loginDomain}/oidc/session/end`,
                            aws_oidc_login_url: `https://${props.loginDomain}`
                        }
                    }
                }
            ],
            dependencies: [cfnCustomResourceRole, cfnCustomResourcePolicy]
        });
        // Deploy site contents to S3 bucket
        new s3deploy.BucketDeployment(this, 'DeployWithInvalidation', {
            sources: [s3deploy.Source.asset('./../portal/build')],
            destinationBucket: websiteBucket,
            distribution: this.website,
            distributionPaths: ['/*'],
            // disable this, otherwise the aws-exports.json will be deleted
            prune: false
        });
    }
    //-------------------------------------------------------
    // Custom Resources Functions
    //-------------------------------------------------------
    addDependencies(resource, dependencies) {
        for (let dependency of dependencies) {
            resource.addDependsOn(dependency);
        }
    }
    createCustomResource(id, customResourceFunction, config) {
        const customResource = new cdk.CfnCustomResource(this, id, {
            serviceToken: customResourceFunction.functionArn
        });
        customResource.addOverride('Type', 'Custom::CustomResource');
        customResource.overrideLogicalId(id);
        if (config) {
            const { properties, condition, dependencies } = config;
            if (properties) {
                for (let property of properties) {
                    customResource.addPropertyOverride(property.path, property.value);
                }
            }
            if (dependencies) {
                this.addDependencies(customResource, dependencies);
            }
            customResource.cfnOptions.condition = condition;
        }
        return customResource;
    }
}
exports.StaticSite = StaticSite;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3RhdGljLXNpdGUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyJzdGF0aWMtc2l0ZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7QUFBQSxzREFBc0Q7QUFDdEQsNERBQTZEO0FBQzdELDJEQUEyRDtBQUMzRCxzQ0FBc0M7QUFDdEMsdURBQXVEO0FBQ3ZELHFDQUFxQztBQUNyQyx3Q0FBd0M7QUFDeEMsd0NBQXdDO0FBQ3hDLDhDQUE4QztBQUM5Qyw2QkFBNEI7QUFtQjVCOzs7OztHQUtHO0FBQ0gsTUFBYSxVQUFXLFNBQVEsZ0JBQVM7SUFHdkMsWUFBWSxNQUFpQixFQUFFLElBQVksRUFBRSxLQUFzQjtRQUNqRSxLQUFLLENBQUMsTUFBTSxFQUFFLElBQUksQ0FBQyxDQUFDO1FBRXBCLE1BQU0sU0FBUyxHQUFHLE1BQU0sQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLFdBQVcsQ0FBQyxDQUFBO1FBRXhELE1BQU0sYUFBYSxHQUFHLElBQUksRUFBRSxDQUFDLE1BQU0sQ0FBQyxJQUFJLEVBQUUsZUFBZSxFQUFFO1lBQ3pELFNBQVMsRUFBRSxLQUFLO1lBQ2hCLGFBQWEsRUFBRSxHQUFHLENBQUMsYUFBYSxDQUFDLE9BQU87WUFDeEMsaUJBQWlCLEVBQUUsSUFBSTtTQUN4QixDQUFDLENBQUM7UUFFSCxhQUFhO1FBQ2IsSUFBSSxDQUFDLE9BQU8sR0FBRyxJQUFJLFVBQVUsQ0FBQyxZQUFZLENBQUMsSUFBSSxFQUFFLHFCQUFxQixFQUFFO1lBQ3RFLGVBQWUsRUFBRTtnQkFDZixNQUFNLEVBQUUsSUFBSSxPQUFPLENBQUMsUUFBUSxDQUFDLGFBQWEsQ0FBQztnQkFDM0Msb0JBQW9CLEVBQUUscUNBQW9CLENBQUMsaUJBQWlCO2FBQzdEO1lBQ0QsaUJBQWlCLEVBQUUsWUFBWTtZQUMvQixHQUFHLENBQUMsU0FBUyxLQUFLLFFBQVEsQ0FBQyxJQUFJO2dCQUM3QixXQUFXLEVBQUUsQ0FBRSxLQUFLLENBQUMsTUFBTSxDQUFFO2dCQUM3QixVQUFV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VSxDQUFDLENBQUM7U0FDbkM7SUFDSCxDQUFDO0lBRUQsb0JBQW9CLENBQUMsRUFBVSxFQUFFLHNCQUF1QyxFQUFFLE1BQTZCO1FBQ3JHLE1BQU0sY0FBYyxHQUFHLElBQUksR0FBRyxDQUFDLGlCQUFpQixDQUFDLElBQUksRUFBRSxFQUFFLEVBQUU7WUFDekQsWUFBWSxFQUFFLHNCQUFzQixDQUFDLFdBQVc7U0FDakQsQ0FBQyxDQUFDO1FBQ0gsY0FBYyxDQUFDLFdBQVcsQ0FBQyxNQUFNLEVBQUUsd0JBQXdCLENBQUMsQ0FBQztRQUM3RCxjQUFjLENBQUMsaUJBQWlCLENBQUMsRUFBRSxDQUFDLENBQUM7UUFDckMsSUFBSSxNQUFNLEVBQUU7WUFDVixNQUFNLEVBQUUsVUFBVSxFQUFFLFNBQVMsRUFBRSxZQUFZLEVBQUUsR0FBRyxNQUFNLENBQUM7WUFDdkQsSUFBSSxVQUFVLEVBQUU7Z0JBQ2QsS0FBSyxJQUFJLFFBQVEsSUFBSSxVQUFVLEVBQUU7b0JBQy9CLGNBQWMsQ0FBQyxtQkFBbUIsQ0FBQyxRQUFRLENBQUMsSUFBSSxFQUFFLFFBQVEsQ0FBQyxLQUFLLENBQUMsQ0FBQztpQkFDbkU7YUFDRjtZQUNELElBQUksWUFBWSxFQUFFO2dCQUNoQixJQUFJLENBQUMsZUFBZSxDQUFDLGNBQWMsRUFBRSxZQUFZLENBQUMsQ0FBQzthQUNwRDtZQUNELGNBQWMsQ0FBQyxVQUFVLENBQUMsU0FBUyxHQUFHLFNBQVMsQ0FBQztTQUNqRDtRQUNELE9BQU8sY0FBYyxDQUFDO0lBQ3hCLENBQUM7Q0FFRjtBQTFNRCxnQ0EwTUMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBjbG91ZGZyb250IGZyb20gJ0Bhd3MtY2RrL2F3cy1jbG91ZGZyb250JztcbmltcG9ydCB7Vmlld2VyUHJvdG9jb2xQb2xpY3l9IGZyb20gJ0Bhd3MtY2RrL2F3cy1jbG91ZGZyb250JztcbmltcG9ydCAqIGFzIG9yaWdpbnMgZnJvbSAnQGF3cy1jZGsvYXdzLWNsb3VkZnJvbnQtb3JpZ2lucyc7XG5pbXBvcnQgKiBhcyBzMyBmcm9tICdAYXdzLWNkay9hd3MtczMnO1xuaW1wb3J0ICogYXMgczNkZXBsb3kgZnJvbSAnQGF3cy1jZGsvYXdzLXMzLWRlcGxveW1lbnQnO1xuaW1wb3J0ICogYXMgY2RrIGZyb20gJ0Bhd3MtY2RrL2NvcmUnO1xuaW1wb3J0IHtDb25zdHJ1Y3R9IGZyb20gJ0Bhd3MtY2RrL2NvcmUnO1xuaW1wb3J0ICogYXMgaWFtIGZyb20gJ0Bhd3MtY2RrL2F3cy1pYW0nO1xuaW1wb3J0ICogYXMgbGFtYmRhIGZyb20gJ0Bhd3MtY2RrL2F3cy1sYW1iZGEnO1xuaW1wb3J0ICogYXMgcGF0aCBmcm9tICdwYXRoJ1xuaW1wb3J0IHsgQXV0aFR5cGUgfSBmcm9tICcuL2luZGV4JztcblxuaW50ZXJmYWNlIEN1c3RvbVJlc291cmNlQ29uZmlnIHtcbiAgcmVhZG9ubHkgcHJvcGVydGllcz86IHsgcGF0aDogc3RyaW5nLCB2YWx1ZTogYW55IH1bXTtcbiAgcmVhZG9ubHkgY29uZGl0aW9uPzogY2RrLkNmbkNvbmRpdGlvbjtcbiAgcmVhZG9ubHkgZGVwZW5kZW5jaWVzPzogY2RrLkNmblJlc291cmNlW107XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgU3RhdGljU2l0ZVByb3BzIHtcbiAgZG9tYWluPzogc3RyaW5nO1xuICAvLyBzc2xDZXJ0aWZpY2F0ZUlhbUlkPzogc3RyaW5nO1xuICBsb2dpbkRvbWFpbj86IHN0cmluZztcbiAgYXBpVXJsOiBzdHJpbmc7XG4gIGF1dGhfdHlwZTogc3RyaW5nO1xuICBhd3NfdXNlcl9wb29sc19pZDogc3RyaW5nO1xuICBhd3NfdXNlcl9wb29sc193ZWJfY2xpZW50X2lkOiBzdHJpbmc7XG59XG5cbi8qKlxuICogU3RhdGljIHNpdGUgaW5mcmFzdHJ1Y3R1cmUsIHdoaWNoIGRlcGxveXMgc2l0ZSBjb250ZW50IHRvIGFuIFMzIGJ1Y2tldC5cbiAqXG4gKiBUaGUgc2l0ZSByZWRpcmVjdHMgZnJvbSBIVFRQIHRvIEhUVFBTLCB1c2luZyBhIENsb3VkRnJvbnQgZGlzdHJpYnV0aW9uLFxuICogUm91dGU1MyBhbGlhcyByZWNvcmQsIGFuZCBBQ00gY2VydGlmaWNhdGUuXG4gKi9cbmV4cG9ydCBjbGFzcyBTdGF0aWNTaXRlIGV4dGVuZHMgQ29uc3RydWN0IHtcbiAgcmVhZG9ubHkgd2Vic2l0ZTogY2xvdWRmcm9udC5EaXN0cmlidXRpb25cblxuICBjb25zdHJ1Y3RvcihwYXJlbnQ6IENvbnN0cnVjdCwgbmFtZTogc3RyaW5nLCBwcm9wczogU3RhdGljU2l0ZVByb3BzKSB7XG4gICAgc3VwZXIocGFyZW50LCBuYW1lKTtcblxuICAgIGNvbnN0IHBhcnRpdGlvbiA9IHBhcmVudC5ub2RlLnRyeUdldENvbnRleHQoJ1BhcnRpdGlvbicpXG5cbiAgICBjb25zdCB3ZWJzaXRlQnVja2V0ID0gbmV3IHMzLkJ1Y2tldCh0aGlzLCAnV2Vic2l0ZUJ1Y2tldCcsIHtcbiAgICAgIHZlcnNpb25lZDogZmFsc2UsXG4gICAgICByZW1vdmFsUG9saWN5OiBjZGsuUmVtb3ZhbFBvbGljeS5ERVNUUk9ZLFxuICAgICAgYXV0b0RlbGV0ZU9iamVjdHM6IHRydWVcbiAgICB9KTtcblxuICAgIC8vIEB0cy1pZ25vcmVcbiAgICB0aGlzLndlYnNpdGUgPSBuZXcgY2xvdWRmcm9udC5EaXN0cmlidXRpb24odGhpcywgJ1dlYnNpdGVEaXN0cmlidXRpb24nLCB7XG4gICAgICBkZWZhdWx0QmVoYXZpb3I6IHtcbiAgICAgICAgb3JpZ2luOiBuZXcgb3JpZ2lucy5TM09yaWdpbih3ZWJzaXRlQnVja2V0KSxcbiAgICAgICAgdmlld2VyUHJvdG9jb2xQb2xpY3k6IFZpZXdlclByb3RvY29sUG9saWN5LlJFRElSRUNUX1RPX0hUVFBTLFxuICAgICAgfSxcbiAgICAgIGRlZmF1bHRSb290T2JqZWN0OiAnaW5kZXguaHRtbCcsXG4gICAgICAuLi4ocGFydGl0aW9uID09PSAnYXdzLWNuJykgJiYge1xuICAgICAgICBkb21haW5OYW1lczogWyBwcm9wcy5kb21haW4gXSxcbiAgICAgICAgZW5hYmxlSXB2NjogZmFsc2UsXG4gICAgICAgIHByaWNlQ2xhc3M6IGNsb3VkZnJvbnQuUHJpY2VDbGFzcy5QUklDRV9DTEFTU19BTExcbiAgICAgIH1cbiAgICB9KVxuXG4gICAgaWYgKHBhcnRpdGlvbiA9PT0gJ2F3cy1jbicpIHtcbiAgICAgIC8qKlxuICAgICAgICogIEdlbmVyYXRlZCBEZWZhdWx0IENhY2hlIEJlaGF2aW9yXG4gICAgICAgKiAgXCJEZWZhdWx0Q2FjaGVCZWhhdmlvclwiOiB7XG4gICAgICAgKiAgICAgICBcIkNhY2hlUG9saWN5SWRcIjogXCI2NTgzMjdlYS1mODlkLTRmYWItYTYzZC03ZTg4NjM5ZTU4ZjZcIixcbiAgICAgICAqICAgICAgIFwiQ29tcHJlc3NcIjogdHJ1ZSxcbiAgICAgICAqICAgICAgIFwiVGFyZ2V0T3JpZ2luSWRcIjogXCJtbGJvdFN0YXRpY1NpdGVDbG91ZEZyb250VG9TM0Nsb3VkRnJvbnREaXN0cmlidXRpb25PcmlnaW4xNzkxM0Q0QzVcIixcbiAgICAgICAqICAgICAgIFwiVmlld2VyUHJvdG9jb2xQb2xpY3lcIjogXCJyZWRpcmVjdC10by1odHRwc1wiXG4gICAgICAgKiAgICAgfSxcbiAgICAgICAqXG4gICAgICAgKiAgIENhY2hlUG9saWN5SWQgaXMgbm90IHN1cHBvcnRlZCBpbiBhd3MtY24gcGFydGl0aW9uXG4gICAgICAgKiAgIEV4cGVjdGVkIERlZmF1bHQgQ2FjaGUgQmVoYXZpb3JcbiAgICAgICAqXG4gICAgICAgKiAgIFwiRGVmYXVsdENhY2hlQmVoYXZpb3JcIjoge1xuICAgICAgICogICAgICAgXCJBbGxvd2VkTWV0aG9kc1wiOiBbIFwiR0VUXCIsIFwiSEVBRFwiIF0sXG4gICAgICAgKiAgICAgICBcIlRhcmdldE9yaWdpbklkXCI6IFwibWxib3RTdGF0aWNTaXRlQ2xvdWRGcm9udFRvUzNDbG91ZEZyb250RGlzdHJpYnV0aW9uT3JpZ2luMTc5MTNENEM1XCIsXG4gICAgICAgKiAgICAgICBcIkZvcndhcmRlZFZhbHVlc1wiOiB7XG4gICAgICAgKiAgICAgICAgICAgXCJRdWVyeVN0cmluZ1wiOiBmYWxzZSxcbiAgICAgICAqICAgICAgICAgICBcIkhlYWRlcnNcIjogWyBcIk9yaWdpblwiLCBcIkFjY2VwdFwiIF0sXG4gICAgICAgKiAgICAgICAgICAgXCJDb29raWVzXCI6IHsgXCJGb3J3YXJkXCI6IFwibm9uZVwiIH1cbiAgICAgICAqICAgICAgIH0sXG4gICAgICAgKiAgICAgICBcIlZpZXdlclByb3RvY29sUG9saWN5XCI6IFwicmVkaXJlY3QtdG8taHR0cHNcIlxuICAgICAgICogICB9LFxuICAgICAgICpcbiAgICAgICAqL1xuICAgICAgY29uc3QgY2ZuRGlzdHJpYnV0aW9uID0gdGhpcy53ZWJzaXRlLm5vZGUuZGVmYXVsdENoaWxkIGFzIGNsb3VkZnJvbnQuQ2ZuRGlzdHJpYnV0aW9uO1xuICAgICAgY2ZuRGlzdHJpYnV0aW9uLmFkZFByb3BlcnR5RGVsZXRpb25PdmVycmlkZSgnRGlzdHJpYnV0aW9uQ29uZmlnLkRlZmF1bHRDYWNoZUJlaGF2aW9yLkNhY2hlUG9saWN5SWQnKTtcbiAgICAgIGNmbkRpc3RyaWJ1dGlvbi5hZGRQcm9wZXJ0eU92ZXJyaWRlKCdEaXN0cmlidXRpb25Db25maWcuRGVmYXVsdENhY2hlQmVoYXZpb3IuQWxsb3dlZE1ldGhvZHMnLCBbIFwiR0VUXCIsIFwiSEVBRFwiIF0pXG4gICAgICBjZm5EaXN0cmlidXRpb24uYWRkUHJvcGVydHlPdmVycmlkZSgnRGlzdHJpYnV0aW9uQ29uZmlnLkRlZmF1bHRDYWNoZUJlaGF2aW9yLkZvcndhcmRlZFZhbHVlcycsIHtcbiAgICAgICAgUXVlcnlTdHJpbmc6IGZhbHNlLFxuICAgICAgICBIZWFkZXJzOiBbIFwiT3JpZ2luXCIsIFwiQWNjZXB0XCIgXSxcbiAgICAgICAgQ29va2llczogeyBGb3J3YXJkOiBcIm5vbmVcIiB9XG4gICAgICB9KVxuXG4gICAgICAvKipcbiAgICAgICAqIGNsb3VkZnJvbnQuRGlzdHJpYnV0aW9uIGRvZXMgbm90IGFsbG93IFNTTCBDZXJ0aWZpY2F0ZSBpbiBJQU0uXG4gICAgICAgKi9cbiAgICAgIC8vIGNmbkRpc3RyaWJ1dGlvbi5hZGRQcm9wZXJ0eU92ZXJyaWRlKCdEaXN0cmlidXRpb25Db25maWcuVmlld2VyQ2VydGlmaWNhdGUnLCB7XG4gICAgICAvLyAgIElhbUNlcnRpZmljYXRlSWQ6IHtcbiAgICAgIC8vICAgICBSZWY6ICdTU0xDZXJ0aWZpY2F0ZUlkJ1xuICAgICAgLy8gICB9LFxuICAgICAgLy8gICBTc2xTdXBwb3J0TWV0aG9kOiAnc25pLW9ubHknLFxuICAgICAgLy8gICBNaW5pbXVtUHJvdG9jb2xWZXJzaW9uOiAnVExTdjEuMl8yMDE4J1xuICAgICAgLy8gfSlcbiAgICB9XG5cbiAgICAvLy0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAvLyBDdXN0b20gUmVzb3VyY2VzXG4gICAgLy8tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgXG4gICAgLy8gQ3VzdG9tUmVzb3VyY2VSb2xlXG4gICAgY29uc3QgY3VzdG9tUmVzb3VyY2VSb2xlID0gbmV3IGlhbS5Sb2xlKHRoaXMsICdDdXN0b21SZXNvdXJjZVJvbGUnLCB7XG4gICAgICBhc3N1bWVkQnk6IG5ldyBpYW0uU2VydmljZVByaW5jaXBhbCgnbGFtYmRhLmFtYXpvbmF3cy5jb20nKSxcbiAgICAgIHBhdGg6ICcvJyxcbiAgICAgIHJvbGVOYW1lOiBgJHtjZGsuQXdzLlNUQUNLX05BTUV9Q3VzdG9tUmVzb3VyY2VSb2xlLSR7Y2RrLkF3cy5SRUdJT059YFxuICAgIH0pXG4gICAgY29uc3QgY2ZuQ3VzdG9tUmVzb3VyY2VSb2xlID0gY3VzdG9tUmVzb3VyY2VSb2xlLm5vZGUuZGVmYXVsdENoaWxkIGFzIGlhbS5DZm5Sb2xlO1xuICAgIGNmbkN1c3RvbVJlc291cmNlUm9sZS5vdmVycmlkZUxvZ2ljYWxJZCgnQ3VzdG9tUmVzb3VyY2VSb2xlJyk7XG5cbiAgICAvLyBDdXN0b21SZXNvdXJjZVBvbGljeVxuICAgIGNvbnN0IGN1c3RvbVJlc291cmNlUG9saWN5ID0gbmV3IGlhbS5Qb2xpY3kodGhpcywgJ0N1c3RvbVJlc291cmNlUG9saWN5Jywge1xuICAgICAgcG9saWN5TmFtZTogYCR7Y2RrLkF3cy5TVEFDS19OQU1FfUN1c3RvbVJlc291cmNlUG9saWN5YCxcbiAgICAgIHN0YXRlbWVudHM6IFtcbiAgICAgICAgbmV3IGlhbS5Qb2xpY3lTdGF0ZW1lbnQoe1xuICAgICAgICAgIGFjdGlvbnM6IFtcbiAgICAgICAgICAgICdsb2dzOkNyZWF0ZUxvZ1N0cmVhbScsXG4gICAgICAgICAgICAnbG9nczpDcmVhdGVMb2dHcm91cCcsXG4gICAgICAgICAgICAnbG9nczpQdXRMb2dFdmVudHMnXG4gICAgICAgICAgXSxcbiAgICAgICAgICByZXNvdXJjZXM6IFtcbiAgICAgICAgICAgIGBhcm46JHtjZGsuQXdzLlBBUlRJVElPTn06bG9nczoke2Nkay5Bd3MuUkVHSU9OfToke2Nkay5Bd3MuQUNDT1VOVF9JRH06bG9nLWdyb3VwOi9hd3MvbGFtYmRhLypgXG4gICAgICAgICAgXVxuICAgICAgICB9KSxcbiAgICAgICAgbmV3IGlhbS5Qb2xpY3lTdGF0ZW1lbnQoe1xuICAgICAgICAgIGFjdGlvbnM6IFsnczM6R2V0T2JqZWN0JywgJ3MzOlB1dE9iamVjdCcsICdzMzpMaXN0QnVja2V0J10sXG4gICAgICAgICAgcmVzb3VyY2VzOiBbYGFybjoke2Nkay5Bd3MuUEFSVElUSU9OfTpzMzo6OipgXVxuICAgICAgICB9KVxuICAgICAgXVxuICAgIH0pO1xuICAgIGN1c3RvbVJlc291cmNlUG9saWN5LmF0dGFjaFRvUm9sZShjdXN0b21SZXNvdXJjZVJvbGUpO1xuICAgIGNvbnN0IGNmbkN1c3RvbVJlc291cmNlUG9saWN5ID0gY3VzdG9tUmVzb3VyY2VQb2xpY3kubm9kZS5kZWZhdWx0Q2hpbGQgYXMgaWFtLkNmblBvbGljeTtcbiAgICBjZm5DdXN0b21SZXNvdXJjZVBvbGljeS5vdmVycmlkZUxvZ2ljYWxJZCgnQ3VzdG9tUmVzb3VyY2VQb2xpY3knKTtcblxuICAgIGNvbnN0IGN1c3RvbVJlc291cmNlRnVuY3Rpb24gPSBuZXcgbGFtYmRhLkZ1bmN0aW9uKHRoaXMsICdDdXN0b21IYW5kbGVyJywge1xuICAgICAgZGVzY3JpcHRpb246ICdBV1MgTUwgQm90IC0gQ3VzdG9tIHJlc291cmNlJyxcbiAgICAgIHJ1bnRpbWU6IGxhbWJkYS5SdW50aW1lLk5PREVKU18xMl9YLFxuICAgICAgaGFuZGxlcjogJ2luZGV4LmhhbmRsZXInLFxuICAgICAgdGltZW91dDogY2RrLkR1cmF0aW9uLnNlY29uZHMoMzApLFxuICAgICAgbWVtb3J5U2l6ZTogNTEyLFxuICAgICAgcm9sZTogY3VzdG9tUmVzb3VyY2VSb2xlLFxuICAgICAgY29kZTogbGFtYmRhLkNvZGUuZnJvbUFzc2V0KHBhdGguam9pbihfX2Rpcm5hbWUsICcuLi9jdXN0b20tcmVzb3VyY2UvJyksIHtcbiAgICAgICAgYnVuZGxpbmc6IHtcbiAgICAgICAgICBpbWFnZTogbGFtYmRhLlJ1bnRpbWUuTk9ERUpTXzEyX1guYnVuZGxpbmdEb2NrZXJJbWFnZSxcbiAgICAgICAgICBjb21tYW5kOiBbXG4gICAgICAgICAgICAnYmFzaCcsICctYycsIFtcbiAgICAgICAgICAgICAgYGNkIC9hc3NldC1vdXRwdXQvYCxcbiAgICAgICAgICAgICAgYGNwIC1yIC9hc3NldC1pbnB1dC8qIC9hc3NldC1vdXRwdXQvYCxcbiAgICAgICAgICAgICAgYGNkIC9hc3NldC1vdXRwdXQvYCxcbiAgICAgICAgICAgICAgYG5wbSBpbnN0YWxsYFxuICAgICAgICAgICAgXS5qb2luKCcgJiYgJylcbiAgICAgICAgICBdLFxuICAgICAgICAgIHVzZXI6ICdyb290J1xuICAgICAgICB9XG4gICAgICB9KVxuICAgIH0pXG5cbiAgICAvLyBDdXN0b21SZXNvdXJjZUNvbmZpZ1xuICAgIHRoaXMuY3JlYXRlQ3VzdG9tUmVzb3VyY2UoJ0N1c3RvbVJlc291cmNlQ29uZmlnJywgY3VzdG9tUmVzb3VyY2VGdW5jdGlvbiwge1xuICAgICAgcHJvcGVydGllczogW1xuICAgICAgICB7IHBhdGg6ICdSZWdpb24nLCB2YWx1ZTogY2RrLkF3cy5SRUdJT04gfSxcbiAgICAgICAgeyBwYXRoOiAnZGVzdFMzQnVja2V0JywgdmFsdWU6IHdlYnNpdGVCdWNrZXQuYnVja2V0TmFtZSB9LFxuICAgICAgICB7IHBhdGg6ICdkZXN0UzNrZXknLCB2YWx1ZTogJ2F3cy1leHBvcnRzLmpzb24nIH0sXG4gICAgICAgIHsgcGF0aDogJ2N1c3RvbUFjdGlvbicsIHZhbHVlOiAncHV0Q29uZmlnRmlsZScgfSxcbiAgICAgICAge1xuICAgICAgICAgIHBhdGg6ICdjb25maWdJdGVtJywgdmFsdWU6IHtcbiAgICAgICAgICAgIGFwaVVybDogcHJvcHMuYXBpVXJsLFxuICAgICAgICAgICAgYXdzX3Byb2plY3RfcmVnaW9uOiBjZGsuQXdzLlJFR0lPTixcbiAgICAgICAgICAgIGF1dGhUeXBlOiBwcm9wcy5hdXRoX3R5cGUgPT09IEF1dGhUeXBlLk9QRU5JRCA/ICdPUEVOSUQnIDogJ0NPR05JVE8nLFxuICAgICAgICAgICAgYXdzX3VzZXJfcG9vbHNfaWQ6IHByb3BzLmF3c191c2VyX3Bvb2xzX2lkLFxuICAgICAgICAgICAgYXdzX3VzZXJfcG9vbHNfd2ViX2NsaWVudF9pZDogcHJvcHMuYXdzX3VzZXJfcG9vbHNfd2ViX2NsaWVudF9pZCxcbiAgICAgICAgICAgIC4uLihwYXJ0aXRpb24gPT09ICdhd3MtY24nKSAmJiB7XG4gICAgICAgICAgICAgIGF3c19vaWRjX3Rva2VuX3ZhbGlkYXRpb25fdXJsOiBgaHR0cHM6Ly8ke3Byb3BzLmxvZ2luRG9tYWlufS9hcGkvdjIvb2lkYy92YWxpZGF0ZV90b2tlbmAsXG4gICAgICAgICAgICAgIGF3c19vaWRjX3Byb3ZpZGVyOiBgaHR0cHM6Ly8ke3Byb3BzLmxvZ2luRG9tYWlufS9vaWRjYCxcbiAgICAgICAgICAgICAgYXdzX29pZGNfbG9nb3V0X3VybDogYGh0dHBzOi8vJHtwcm9wcy5sb2dpbkRvbWFpbn0vb2lkYy9zZXNzaW9uL2VuZGAsXG4gICAgICAgICAgICAgIGF3c19vaWRjX2xvZ2luX3VybDogYGh0dHBzOi8vJHtwcm9wcy5sb2dpbkRvbWFpbn1gXG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICBdLFxuICAgICAgZGVwZW5kZW5jaWVzOiBbIGNmbkN1c3RvbVJlc291cmNlUm9sZSwgY2ZuQ3VzdG9tUmVzb3VyY2VQb2xpY3kgXVxuICAgIH0pO1xuXG4gICAgLy8gRGVwbG95IHNpdGUgY29udGVudHMgdG8gUzMgYnVja2V0XG4gICAgbmV3IHMzZGVwbG95LkJ1Y2tldERlcGxveW1lbnQodGhpcywgJ0RlcGxveVdpdGhJbnZhbGlkYXRpb24nLCB7XG4gICAgICBzb3VyY2VzOiBbIHMzZGVwbG95LlNvdXJjZS5hc3NldCgnLi8uLi9wb3J0YWwvYnVpbGQnKSBdLFxuICAgICAgZGVzdGluYXRpb25CdWNrZXQ6IHdlYnNpdGVCdWNrZXQsXG4gICAgICBkaXN0cmlidXRpb246IHRoaXMud2Vic2l0ZSxcbiAgICAgIGRpc3RyaWJ1dGlvblBhdGhzOiBbJy8qJ10sXG4gICAgICAvLyBkaXNhYmxlIHRoaXMsIG90aGVyd2lzZSB0aGUgYXdzLWV4cG9ydHMuanNvbiB3aWxsIGJlIGRlbGV0ZWRcbiAgICAgIHBydW5lOiBmYWxzZVxuICAgIH0pO1xuICB9XG5cbiAgLy8tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gIC8vIEN1c3RvbSBSZXNvdXJjZXMgRnVuY3Rpb25zXG4gIC8vLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuXG4gIGFkZERlcGVuZGVuY2llcyhyZXNvdXJjZTogY2RrLkNmblJlc291cmNlLCBkZXBlbmRlbmNpZXM6IGNkay5DZm5SZXNvdXJjZVtdKSB7XG4gICAgZm9yIChsZXQgZGVwZW5kZW5jeSBvZiBkZXBlbmRlbmNpZXMpIHtcbiAgICAgIHJlc291cmNlLmFkZERlcGVuZHNPbihkZXBlbmRlbmN5KTtcbiAgICB9XG4gIH1cblxuICBjcmVhdGVDdXN0b21SZXNvdXJjZShpZDogc3RyaW5nLCBjdXN0b21SZXNvdXJjZUZ1bmN0aW9uOiBsYW1iZGEuRnVuY3Rpb24sIGNvbmZpZz86IEN1c3RvbVJlc291cmNlQ29uZmlnKTogY2RrLkNmbkN1c3RvbVJlc291cmNlIHtcbiAgICBjb25zdCBjdXN0b21SZXNvdXJjZSA9IG5ldyBjZGsuQ2ZuQ3VzdG9tUmVzb3VyY2UodGhpcywgaWQsIHtcbiAgICAgIHNlcnZpY2VUb2tlbjogY3VzdG9tUmVzb3VyY2VGdW5jdGlvbi5mdW5jdGlvbkFyblxuICAgIH0pO1xuICAgIGN1c3RvbVJlc291cmNlLmFkZE92ZXJyaWRlKCdUeXBlJywgJ0N1c3RvbTo6Q3VzdG9tUmVzb3VyY2UnKTtcbiAgICBjdXN0b21SZXNvdXJjZS5vdmVycmlkZUxvZ2ljYWxJZChpZCk7XG4gICAgaWYgKGNvbmZpZykge1xuICAgICAgY29uc3QgeyBwcm9wZXJ0aWVzLCBjb25kaXRpb24sIGRlcGVuZGVuY2llcyB9ID0gY29uZmlnO1xuICAgICAgaWYgKHByb3BlcnRpZXMpIHtcbiAgICAgICAgZm9yIChsZXQgcHJvcGVydHkgb2YgcHJvcGVydGllcykge1xuICAgICAgICAgIGN1c3RvbVJlc291cmNlLmFkZFByb3BlcnR5T3ZlcnJpZGUocHJvcGVydHkucGF0aCwgcHJvcGVydHkudmFsdWUpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBpZiAoZGVwZW5kZW5jaWVzKSB7XG4gICAgICAgIHRoaXMuYWRkRGVwZW5kZW5jaWVzKGN1c3RvbVJlc291cmNlLCBkZXBlbmRlbmNpZXMpO1xuICAgICAgfVxuICAgICAgY3VzdG9tUmVzb3VyY2UuY2ZuT3B0aW9ucy5jb25kaXRpb24gPSBjb25kaXRpb247XG4gICAgfVxuICAgIHJldHVybiBjdXN0b21SZXNvdXJjZTtcbiAgfVxuXG59XG4iXX0=