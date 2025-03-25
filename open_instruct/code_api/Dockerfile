# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY open_instruct/code_api/code_api_requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY open_instruct/code_api/ .

# Create cache directory for code execution
RUN mkdir -p cache && chmod 777 cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose the port the app runs on
EXPOSE ${PORT}

# Command to run the application
CMD uvicorn api:app --host 0.0.0.0 --port $PORT