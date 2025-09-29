# Use a slim Python image for a smaller container
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
# Install with no-cache to reduce image size
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the project structure directories
RUN mkdir -p ml api tests

# Copy the trained model and encoder into the ml directory
# NOTE: Ensure these files are present in the build context when running 'docker build'
COPY intent_classifier_pipeline.pkl ml/
COPY intent_label_encoder.pkl ml/

# Copy the rest of the application code
COPY api/ api/
COPY ml/model_loader.py ml/
COPY tests/ tests/

# Set the environment variable for Python
ENV PYTHONUNBUFFERED 1

# Expose the port uvicorn runs on
EXPOSE 8000

# Command to run the application with Uvicorn
# The command starts the server which triggers model loading via api.main:app's lifespan
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]