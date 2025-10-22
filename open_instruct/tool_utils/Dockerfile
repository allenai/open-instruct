# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.8 /uv /bin/uv

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements.txt

# Install dependencies using uv
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the application
COPY . .

# Create cache directory for code execution
RUN mkdir -p cache && chmod 777 cache

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["uv", "run", "--no-project", "tool_server.py"]