# Use an official Python runtime as a parent image
FROM python:3.14-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install specific dependencies for Phase 3 Foundation Expert
RUN pip install --no-cache-dir peft transformers accelerate datasets

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the dashboard by default
ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
