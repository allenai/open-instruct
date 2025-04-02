terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-west-2"
  profile = "153242493257_AllenNLP-User"
}

# Use the pre-built Lambda package from Makefile
# Comment out the archive_file resource since the Makefile builds the package
# data "archive_file" "lambda_zip" {
#   type        = "zip"
#   source_dir  = "${path.module}/.."
#   output_path = "${path.module}/lambda_function.zip"
# }

# Local variable for the lambda zip file path
locals {
  lambda_zip_path = "${path.module}/lambda_function.zip"
}

# Create S3 bucket for Lambda code
resource "aws_s3_bucket" "lambda_bucket" {
  bucket_prefix = "test-program-lambda-code-"
}

# Upload Lambda zip to S3
resource "aws_s3_object" "lambda_code" {
  bucket = aws_s3_bucket.lambda_bucket.id
  key    = "lambda_function.zip"
  source = local.lambda_zip_path
  etag   = filemd5(local.lambda_zip_path)
}

# Create the Lambda function
resource "aws_lambda_function" "test_program" {
  function_name    = "test_program"
  role            = "arn:aws:iam::153242493257:role/lambda_basic_execution"
  handler         = "lambda_function.lambda_handler"
  runtime         = "python3.9"
  timeout         = 15
  memory_size     = 256
  
  # Use S3 instead of direct upload
  s3_bucket       = aws_s3_bucket.lambda_bucket.id
  s3_key          = aws_s3_object.lambda_code.key
  source_code_hash = filebase64sha256(local.lambda_zip_path)

  environment {
    variables = {
      PYTHONPATH = "/var/task"
    }
  }
}

# Create API Gateway
resource "aws_apigatewayv2_api" "lambda_api" {
  name          = "test_program_api"
  protocol_type = "HTTP"
  cors_configuration {
    allow_headers = ["*"]
    allow_methods = ["*"]
    allow_origins = ["*"]
    expose_headers = ["*"]
    max_age = 86400
  }
}

# Create API Gateway integration
resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id           = aws_apigatewayv2_api.lambda_api.id
  integration_type = "AWS_PROXY"

  integration_method = "POST"
  integration_uri    = aws_lambda_function.test_program.invoke_arn
}

# Create API Gateway route
resource "aws_apigatewayv2_route" "lambda_route" {
  api_id    = aws_apigatewayv2_api.lambda_api.id
  route_key = "POST /test_program"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
  
  # Add this to prevent caching
  api_key_required = false
  authorization_type = "NONE"
}

# Create API Gateway stage
resource "aws_apigatewayv2_stage" "lambda_stage" {
  api_id = aws_apigatewayv2_api.lambda_api.id
  name   = "prod"

  default_route_settings {
    throttling_burst_limit = 200
    throttling_rate_limit  = 0
  }
}

# Create Lambda permission for API Gateway
resource "aws_lambda_permission" "api_gw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.test_program.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.lambda_api.execution_arn}/*/*"
}

# Output the API endpoint URL
output "api_endpoint" {
  value = "${aws_apigatewayv2_api.lambda_api.api_endpoint}/prod/test_program"
} 