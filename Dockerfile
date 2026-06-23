# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies required for OpenCV and ML libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the InsightFace buffalo_l model weights so they are baked into the image.
# This prevents it from downloading on the first request, which often causes a timeout on Render.
RUN python -c "import os; from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))" || true

# Copy the rest of the application code
COPY . .

# Ensure the port variable is exposed
EXPOSE 8000

# Command to run the FastAPI application. Render will automatically set the PORT environment variable if needed.
# We'll default to 8000 if not set.
CMD uvicorn fastapi_app:app --host 0.0.0.0 --port ${PORT:-8000}
