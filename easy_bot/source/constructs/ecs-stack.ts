import * as ecr from '@aws-cdk/aws-ecr';
import * as ecs from '@aws-cdk/aws-ecs';
import * as ec2 from '@aws-cdk/aws-ec2';
import * as cdk from "@aws-cdk/core"
import { Bucket } from '@aws-cdk/aws-s3';
import { PolicyStatement } from "@aws-cdk/aws-iam"
import { Vpc } from "@aws-cdk/aws-ec2"
import * as ecrAssets  from "@aws-cdk/aws-ecr-assets";
import * as path from 'path';

export interface EcsStackProps {
  dataBucket: Bucket,
  modelBucket: Bucket,
  vpc: ec2.IVpc,
  version: string,
  fileSystemId: string,
  accessPointId: string,
  ec2SecurityGroup: ec2.SecurityGroup,
}

export class EcsStack extends cdk.Construct {
  readonly cluster: ecs.Cluster
  readonly trainingTaskDef: ecs.Ec2TaskDefinition
  readonly inferenceTaskDef: ecs.Ec2TaskDefinition
  readonly trainingAsgName : string
  readonly inferenceAsgName : string
  readonly trainingSG : ec2.SecurityGroup
  readonly inferenceSG : ec2.SecurityGroup

  constructor(parent: cdk.Construct, id: string, props: EcsStackProps) {
    super(parent, id);

    const gpuAmi = new ecs.EcsOptimizedAmi({hardwareType: ecs.AmiHardwareType.GPU});

    // configure security group for training host machine
    this.trainingSG = new ec2.SecurityGroup(this, "TrainingEndpointSG", {
      vpc: props.vpc,
      allowAllOutbound: true,
    });
    this.trainingSG.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(2049));

    // configure security group for inference host machine
    this.inferenceSG = new ec2.SecurityGroup(this, "InferenceEndpointSG", {
      vpc: props.vpc,
      allowAllOutbound: true,
    });
    this.inferenceSG.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(8080));

    const ECR_ORIGIN = parent.node.tryGetContext('EcrOrigin');
    const partition = parent.node.tryGetContext('Partition');

    // TODO: replace the following account ID and region code.
    const baseImageUri = (partition === 'aws-cn') ?
      '727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn' :
      '763104351884.dkr.ecr.us-west-2.amazonaws.com'
    const upstreamRepoArn = (partition === 'aws-cn') ?
      `arn:aws-cn:ecr:${cdk.Aws.REGION}:753680513547:repository` :
      `arn:aws:ecr:${cdk.Aws.REGION}:366590864501:repository`
    
    const trainingRepoName = 'ml-bot-training'
    const inferenceRepoName = 'ml-bot-inference'

    // Construct training image for task definition
    const trainingImageAsset = new ecrAssets.DockerImageAsset(this, 'TrainingImage', {
      directory: path.join(__dirname, '../docker/training/'),
      buildArgs: {
        REGISTRY_URI: baseImageUri
      },
      repositoryName: trainingRepoName
    })
    const trainingImageRepo = ecr.Repository.fromRepositoryAttributes(this, trainingRepoName, {
      repositoryArn: `${upstreamRepoArn}/${trainingRepoName}`,
      repositoryName: trainingRepoName
    })
    const trainingImage = (ECR_ORIGIN && ECR_ORIGIN === 'asset') ?
      ecs.ContainerImage.fromDockerImageAsset(trainingImageAsset) :
      ecs.ContainerImage.fromEcrRepository(trainingImageRepo, props.version)

    // Construct inference image for task definition
    const inferenceImageAsset = new ecrAssets.DockerImageAsset(this, 'InferenceImage', {
      directory: path.join(__dirname, '../docker/inference/'),
      buildArgs: {
        REGISTRY_URI: baseImageUri
      },
      repositoryName: inferenceRepoName
    })
    const inferenceImageRepo = ecr.Repository.fromRepositoryAttributes(this, inferenceRepoName, {
      repositoryArn: `${upstreamRepoArn}/${inferenceRepoName}`,
      repositoryName: inferenceRepoName
    })
    const inferenceImage = (ECR_ORIGIN && ECR_ORIGIN === 'asset') ?
      ecs.ContainerImage.fromDockerImageAsset(inferenceImageAsset) :
      ecs.ContainerImage.fromEcrRepository(inferenceImageRepo, props.version)

    // Create a cluster
    this.cluster = new ecs.Cluster(this, 'Cluster', { vpc: props.vpc });
    const trainingAsg = this.cluster.addCapacity('TrainingASG', {
      minCapacity: 0,
      maxCapacity: 1,
      desiredCapacity: 0,
      instanceType: new ec2.InstanceType('p3.2xlarge'),
      machineImage: gpuAmi,
      associatePublicIpAddress: false,
    });
    const inferenceAsg = this.cluster.addCapacity('InferenceASG', {
      minCapacity: 0,
      maxCapacity: 1,
      desiredCapacity: 0,
      instanceType: new ec2.InstanceType('g4dn.xlarge'),
      machineImage: gpuAmi,
      associatePublicIpAddress: false,
    });

    const s3Policy = new PolicyStatement()
    s3Policy.addActions("s3:*")
    s3Policy.addResources(props.dataBucket.bucketArn)
    s3Policy.addResources(props.dataBucket.bucketArn + "/*")
    s3Policy.addResources(props.modelBucket.bucketArn)
    s3Policy.addResources(props.modelBucket.bucketArn + "/*")

    const ecrPolicy = new PolicyStatement()
    ecrPolicy.addActions("ecr:*")
    ecrPolicy.addResources("*")

    const efsPolicy = new PolicyStatement()
    efsPolicy.addActions("elasticfilesystem:*")
    efsPolicy.addResources("*")

    trainingAsg.addToRolePolicy(s3Policy)
    trainingAsg.addToRolePolicy(ecrPolicy)
    trainingAsg.addToRolePolicy(efsPolicy)
    trainingAsg.addSecurityGroup(props.ec2SecurityGroup)
    trainingAsg.addUserData(
      "sudo yum install -y amazon-efs-utils",
      "sudo systemctl enable --now amazon-ecs-volume-plugin",
      "sudo mkdir -p  /mnt/ml",
      "sudo chmod go+rw /mnt/ml",
      `sudo mount -t efs -o tls,accesspoint=${props.accessPointId} ${props.fileSystemId}:/ /mnt/ml`
    )
    trainingAsg.addSecurityGroup(this.trainingSG)
    inferenceAsg.addToRolePolicy(s3Policy)
    inferenceAsg.addToRolePolicy(ecrPolicy)
    inferenceAsg.addSecurityGroup(this.inferenceSG)
    
    // create a task definition with CloudWatch Logs
    const logging = new ecs.AwsLogDriver({ streamPrefix: "ml-bot" })
    const linuxParameters = new ecs.LinuxParameters(this, "LinuxParameters", {
      sharedMemorySize: 2048,
    });

    this.trainingTaskDef = new ecs.Ec2TaskDefinition(this, "TrainingTask");
    const trainingContainerDef = this.trainingTaskDef.addContainer("trainingContainer", {
      image: trainingImage,
      gpuCount: 1,
      memoryLimitMiB: 31500,
      logging: logging,
      environment: {
        DATA_BUCKET: props.dataBucket.bucketName,
        DATA_PREFIX: "",
        MODEL_BUCKET: props.modelBucket.bucketName,
        MODEL_PREFIX: "",
        AWS_DEFAULT_REGION: cdk.Aws.REGION,
      },
      linuxParameters: linuxParameters,
    })

    this.trainingTaskDef.addToExecutionRolePolicy(s3Policy)
    this.trainingTaskDef.addToExecutionRolePolicy(ecrPolicy)
    this.trainingTaskDef.addToExecutionRolePolicy(efsPolicy)
    this.trainingTaskDef.addToTaskRolePolicy(s3Policy)
    this.trainingTaskDef.addToTaskRolePolicy(ecrPolicy)
    this.trainingTaskDef.addToTaskRolePolicy(efsPolicy)

    this.trainingTaskDef.addVolume({
      name: "efs-model",
      host: {
        sourcePath: "/mnt/ml"
      }
    })
    trainingContainerDef.addMountPoints({
      containerPath: "/mnt/ml",
      readOnly: false,
      sourceVolume: "efs-model"
    })

    this.inferenceTaskDef = new ecs.Ec2TaskDefinition(this, "InferenceTask");
    this.inferenceTaskDef.addContainer("inferenceContainer", {
      image: inferenceImage,
      gpuCount: 1,
      memoryLimitMiB: 15500,
      logging,
      environment: {
        MODEL_BUCKET: props.modelBucket.bucketName,
        MODEL_PREFIX: "",
        AWS_DEFAULT_REGION: cdk.Aws.REGION,
        MXNET_CUDNN_AUTOTUNE_DEFAULT: "0",
      },
      linuxParameters: linuxParameters,
    }).addPortMappings(
      { containerPort: 8080, hostPort: 8080, protocol: ecs.Protocol.TCP }
    )

    this.inferenceTaskDef.addToExecutionRolePolicy(s3Policy)
    this.inferenceTaskDef.addToExecutionRolePolicy(ecrPolicy)
    this.inferenceTaskDef.addToTaskRolePolicy(s3Policy)
    this.inferenceTaskDef.addToTaskRolePolicy(ecrPolicy)

    this.trainingAsgName = trainingAsg.autoScalingGroupName
    this.inferenceAsgName = inferenceAsg.autoScalingGroupName
  }
}

