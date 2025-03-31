.PHONY: build-lambda deploy-lambda clean-lambda init-terraform plan-lambda destroy-lambda

# Variables
PYTHON = python3
VENV = .venv
LAMBDA_ZIP = terraform/lambda_function.zip
TERRAFORM_DIR = terraform

# Build the Lambda function package
build-lambda: clean-lambda
	@echo "Building Lambda function package..."
	@mkdir -p $(VENV)
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -r requirements.txt
	@echo "Creating Lambda deployment package..."
	@mkdir -p $(TERRAFORM_DIR)
	@cd $(VENV)/lib/python*/site-packages && zip -r ../../../../$(LAMBDA_ZIP) .
	@zip -g $(LAMBDA_ZIP) lambda_function.py __init__.py
	@echo "Lambda package created at $(LAMBDA_ZIP)"

# Initialize Terraform
init-terraform:
	@echo "Initializing Terraform..."
	@cd $(TERRAFORM_DIR) && terraform init

# Plan Terraform changes
plan-lambda:
	@echo "Planning Terraform changes..."
	@cd $(TERRAFORM_DIR) && terraform plan

# Deploy the Lambda function
deploy-lambda: build-lambda init-terraform
	@echo "Deploying Lambda function..."
	@cd $(TERRAFORM_DIR) && terraform apply -auto-approve
	@echo "Deployment complete! API endpoint:"
	@cd $(TERRAFORM_DIR) && terraform output api_endpoint

# Clean up build artifacts
clean-lambda:
	@echo "Cleaning up build artifacts..."
	@rm -rf $(VENV)
	@rm -f $(LAMBDA_ZIP)

# Destroy deployed resources
destroy-lambda:
	@echo "Destroying deployed resources..."
	@cd $(TERRAFORM_DIR) && terraform destroy -auto-approve

# Show help
help:
	@echo "Available commands:"
	@echo "  build-lambda    - Build the Lambda function package"
	@echo "  deploy-lambda   - Deploy the Lambda function to AWS"
	@echo "  clean-lambda    - Clean up build artifacts"
	@echo "  init-terraform  - Initialize Terraform"
	@echo "  plan-lambda     - Show planned Terraform changes"
	@echo "  destroy-lambda  - Destroy deployed resources"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Note: AWS credentials should be configured in your environment"
	@echo "      (e.g., via aws configure or environment variables)" 