import * as cloudfront from '@aws-cdk/aws-cloudfront';
import {ViewerProtocolPolicy} from '@aws-cdk/aws-cloudfront';
import * as origins from '@aws-cdk/aws-cloudfront-origins';
import * as s3 from '@aws-cdk/aws-s3';
import * as s3deploy from '@aws-cdk/aws-s3-deployment';
import * as cdk from '@aws-cdk/core';
import {Construct} from '@aws-cdk/core';
import * as iam from '@aws-cdk/aws-iam';
import * as lambda from '@aws-cdk/aws-lambda';
import * as path from 'path'
import { AuthType } from './index';

interface CustomResourceConfig {
  readonly properties?: { path: string, value: any }[];
  readonly condition?: cdk.CfnCondition;
  readonly dependencies?: cdk.CfnResource[];
}

export interface StaticSiteProps {
  domain?: string;
  // sslCertificateIamId?: string;
  loginDomain?: string;
  apiUrl: string;
  auth_type: string;
  aws_user_pools_id: string;
  aws_user_pools_web_client_id: string;
}

/**
 * Static site infrastructure, which deploys site content to an S3 bucket.
 *
 * The site redirects from HTTP to HTTPS, using a CloudFront distribution,
 * Route53 alias record, and ACM certificate.
 */
export class StaticSite extends Construct {
  readonly website: cloudfront.Distribution

  constructor(parent: Construct, name: string, props: StaticSiteProps) {
    super(parent, name);

    const partition = parent.node.tryGetContext('Partition')

    const websiteBucket = new s3.Bucket(this, 'WebsiteBucket', {
      versioned: false,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true
    });

    // @ts-ignore
    this.website = new cloudfront.Distribution(this, 'WebsiteDistribution', {
      defaultBehavior: {
        origin: new origins.S3Origin(websiteBucket),
        viewerProtocolPolicy: ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
      },
      defaultRootObject: 'index.html',
      ...(partition === 'aws-cn') && {
        domainNames: [ props.domain ],
        enableIpv6: false,
        priceClass: cloudfront.PriceClass.PRICE_CLASS_ALL
      }
    })

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
      const cfnDistribution = this.website.node.defaultChild as cloudfront.CfnDistribution;
      cfnDistribution.addPropertyDeletionOverride('DistributionConfig.DefaultCacheBehavior.CachePolicyId');
      cfnDistribution.addPropertyOverride('DistributionConfig.DefaultCacheBehavior.AllowedMethods', [ "GET", "HEAD" ])
      cfnDistribution.addPropertyOverride('DistributionConfig.DefaultCacheBehavior.ForwardedValues', {
        QueryString: false,
        Headers: [ "Origin", "Accept" ],
        Cookies: { Forward: "none" }
      })

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
    })
    const cfnCustomResourceRole = customResourceRole.node.defaultChild as iam.CfnRole;
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
    const cfnCustomResourcePolicy = customResourcePolicy.node.defaultChild as iam.CfnPolicy;
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
    })

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
            authType: props.auth_type === AuthType.OPENID ? 'OPENID' : 'COGNITO',
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
      dependencies: [ cfnCustomResourceRole, cfnCustomResourcePolicy ]
    });

    // Deploy site contents to S3 bucket
    new s3deploy.BucketDeployment(this, 'DeployWithInvalidation', {
      sources: [ s3deploy.Source.asset('./../portal/build') ],
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

  addDependencies(resource: cdk.CfnResource, dependencies: cdk.CfnResource[]) {
    for (let dependency of dependencies) {
      resource.addDependsOn(dependency);
    }
  }

  createCustomResource(id: string, customResourceFunction: lambda.Function, config?: CustomResourceConfig): cdk.CfnCustomResource {
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
