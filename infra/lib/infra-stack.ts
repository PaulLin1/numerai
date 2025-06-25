import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as path from 'path';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as glue from 'aws-cdk-lib/aws-glue';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as apigateway from "aws-cdk-lib/aws-apigateway";

export class InfraStack extends cdk.Stack {
	constructor(scope: Construct, id: string, props?: cdk.StackProps) {
		super(scope, id, props);

		// Bucket with pretty much everything
		const sageMakerBucket = new s3.Bucket(this, 'NumeraiSageMakerBucket', {
			versioned: true,
			removalPolicy: cdk.RemovalPolicy.DESTROY,
		})

		// Role that has full access to sagemaker and s3
		const sageMakerRole = new iam.Role(this, 'SageMakerRole', {
			assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
			managedPolicies: [
			iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
			iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'),
			],
		});

		// Create a policy to allow ecr:BatchGetImage for the specific ECR repository
		// For xgboost img
		const ecrPolicy = new iam.Policy(this, 'EcrAccessPolicy', {
			statements: [
				new iam.PolicyStatement({
					actions: ['ecr:BatchGetImage', 'ecr:GetAuthorizationToken'],
					resources: [
						'arn:aws:ecr:us-east-2:811284229777:repository/sagemaker-xgboost',
					],
				}),
			],
		});
		ecrPolicy.attachToRole(sageMakerRole);


		// --------------------------------------------------------------------

		// Create the model
		const trainFunction = new lambda.Function(this, 'TrainFunction', {
			runtime: lambda.Runtime.PYTHON_3_10,
			handler: 'handler.lambda_handler',
			code: lambda.Code.fromAsset(this.node.tryGetContext("TRAIN_LAMBDA_DIR")),
			timeout: cdk.Duration.minutes(15),
			environment: {
				SAGEMAKER_ARN: sageMakerRole.roleArn,
				S3_NAME: sageMakerBucket.bucketName
			}
		});

		// Create the model endpoint
		const createEndpointFunction = new lambda.Function(this, 'CreateEndpointFunction', {
			runtime: lambda.Runtime.PYTHON_3_10,
			handler: 'handler.lambda_handler',
			code: lambda.Code.fromAsset(this.node.tryGetContext("ENDPOINT_LAMBDA_DIR")),
			environment: {
				SAGEMAKER_ARN: sageMakerRole.roleArn,
				S3_NAME: sageMakerBucket.bucketName,
			},
			timeout: cdk.Duration.minutes(15),
		});
		
		const sageMakerPolicy = new iam.PolicyStatement({
			actions: ['sagemaker:CreateTrainingJob',
				'sagemaker:CreateModel',
				'sagemaker:CreateEndpoint',
				'sagemaker:CreateEndpointConfig',
				'sagemaker:DescribeEndpoint',
				'sagemaker:DescribeEndpointConfig',
				'sagemaker:DescribeModel',
				'sagemaker:InvokeEndpoint'
			],
			
			resources: ['*'],
		  });
	  
		trainFunction.addToRolePolicy(sageMakerPolicy);
		trainFunction.addToRolePolicy(new iam.PolicyStatement({
			actions: ['iam:PassRole'],
			resources: [sageMakerRole.roleArn],
		}));

		createEndpointFunction.addToRolePolicy(sageMakerPolicy);
		createEndpointFunction.addToRolePolicy(new iam.PolicyStatement({
			actions: ['iam:PassRole'],
			resources: [sageMakerRole.roleArn],
		}));

		// Step function for traning and deploying
		const trainAndDeployStateMachine = new sfn.StateMachine(this, 'TrainAndDeployModel', {
			definition: new tasks.LambdaInvoke(this, 'TrainModelTask', {
				lambdaFunction: trainFunction,
				outputPath: '$.Payload',
			}).next(new sfn.Wait(this, 'WaitForTrainingCompletion', {
				time: sfn.WaitTime.duration(cdk.Duration.minutes(5)),
			})).next(new tasks.LambdaInvoke(this, 'CreateEndpointTask', {
				lambdaFunction: createEndpointFunction,
				payload: sfn.TaskInput.fromObject({
					training_job_name: sfn.JsonPath.stringAt("$.training_job_name")
				}),
				outputPath: '$.Payload',
			})),
			timeout: cdk.Duration.minutes(30),
		});


		// --------------------------------------------------------------------
	  
		const glueJobRole = new iam.Role(this, 'GlueJobRole', {
			assumedBy: new iam.ServicePrincipal('glue.amazonaws.com'),
			managedPolicies: [
				iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSGlueServiceRole')
			]
		});
	
		glueJobRole.addToPolicy(new iam.PolicyStatement({
			actions: [
				's3:GetObject',
				's3:PutObject'
			],
			resources: [sageMakerBucket.bucketArn,
				`${sageMakerBucket.bucketArn}/*`
			]
		}));

		// Glue job for validation
		const glueJob = new glue.CfnJob(this, 'ValidationJob', {
			name: 'sagemaker-validation',
			role: glueJobRole.roleArn,
			command: {
				name: 'glueetl',
				scriptLocation: `s3://${sageMakerBucket.bucketName}/scripts/validation.py`,
				pythonVersion: '3',
			},
			defaultArguments: {
				'--input_path': `s3://${sageMakerBucket.bucketName}/data/v5.0/validation/validation.parquet`,
				'--output_path': `s3://${sageMakerBucket.bucketName}/data/v5.0/validation/`,
				// '--sagemaker_endpoint_name': '/numerai/current_endpoint',
			},
			glueVersion: '3.0',
			maxRetries: 1,
			timeout: 60,
		});


		// --------------------------------------------------------------------

		// Download Live Function
		const downloadLiveFunction = new lambda.Function(this, 'DownloadLiveFunction', {
			runtime: lambda.Runtime.PYTHON_3_12,
			handler: 'handler.lambda_handler',
			code: lambda.Code.fromAsset(this.node.tryGetContext("DOWNLOAD_LIVE_LAMBDA_DIR")),
			timeout: cdk.Duration.minutes(15),
			architecture: lambda.Architecture.ARM_64,
			environment: {
				// SAGEMAKER_ARN: sageMakerRole.roleArn,
				S3_NAME: sageMakerBucket.bucketName
			}
		});
		sageMakerBucket.grantPut(downloadLiveFunction)

		// Predict Function
		const predictFunction = new lambda.Function(this, 'PredictFunction', {
			runtime: lambda.Runtime.PYTHON_3_12,
			handler: 'handler.lambda_handler',
			code: lambda.Code.fromAsset(this.node.tryGetContext("PREDICT_LAMBDA_DIR")),
			timeout: cdk.Duration.minutes(15),
			environment: {
				SAGEMAKER_ARN: sageMakerRole.roleArn,
				S3_NAME: sageMakerBucket.bucketName
			}
		});

		// Grant Lambda permission to invoke SageMaker endpoint
		// predictFunction.addToRolePolicy(
		// 	new iam.PolicyStatement({
		// 	actions: ["sagemaker:InvokeEndpoint"],
		// 	resources: ["arn:aws:sagemaker:us-east-1:your-account-id:endpoint/your-xgboost-endpoint-name"],
		// 	})
		// );
	}
}
