# Lambda Deployment Instructions

This guide explains how to deploy the test_program endpoint as an AWS Lambda function.

## Prerequisites

1. AWS CLI installed and configured with appropriate credentials
2. Terraform installed
3. Python 3.9 or later
4. `zip` utility installed (on Linux: `sudo apt-get install zip`)

## AWS Credentials Setup

Make sure you have AWS credentials configured in your environment. You can do this by adding the following to your `~/.aws/credentials` file:

[153242493257_AllenNLP-User]
aws_access_key_id=<YOUR_KEY>
aws_secret_access_key=<YOUR_KEY>
aws_session_token=<YOUR_TOKEN>

## Deployment Steps

1. Build and deploy the Lambda function with:
```bash
make deploy-lambda
```

This command will:
- Create a virtual environment
- Install dependencies
- Build the Lambda package
- Initialize Terraform
- Deploy the function to AWS
- Output the API endpoint URL

## Alternative Commands

If you need more control over the deployment process, you can use these individual commands:

1. Build the Lambda package without deploying:
```bash
make build-lambda
```

2. View planned Terraform changes:
```bash
make plan-lambda
```

3. Clean up build artifacts:
```bash
make clean-lambda
```

4. Remove all deployed resources:
```bash
make destroy-lambda
```

## Testing the Lambda Function

You can test the deployed Lambda function using curl or any HTTP client:

```bash
curl -X POST https://your-api-endpoint/test_program \
  -H "Content-Type: application/json" \
  -d '{
    "program": "def add(a, b):\n    return a + b",
    "tests": [
      "assert add(1, 2) == 3",
      "assert add(-1, 1) == 0"
    ],
    "max_execution_time": 1.0
  }'
```

## Notes

- The Lambda function has a 30-second timeout and 256MB memory allocation
- The function is deployed with CORS enabled
- The API Gateway endpoint is publicly accessible
- Make sure to update the AWS region in `terraform/main.tf` if needed 