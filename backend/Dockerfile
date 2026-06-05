# Use an official Node.js runtime as a parent image
FROM node:18-slim

# Install Python 3, pip, virtual environment, and dependencies for OpenCV/InsightFace
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json first to leverage Docker cache
COPY package*.json ./

# Install Node.js dependencies
RUN npm install --omit=dev

# Copy the Python requirements file
COPY requirements.txt ./

# Set up Python virtual environment and install dependencies
RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the API port
EXPOSE 5001

# Command to run the application
CMD ["node", "server.js"]
