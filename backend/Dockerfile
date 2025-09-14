# Use Python 3.10 slim base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt . 

# Make sure pip is updated
RUN pip install --upgrade pip

# Install requirements + redis client
RUN pip install -r requirements.txt && pip install redis

# Copy your backend code into the container
COPY . .

# Copy your ML model explicitly (if required)
COPY stock_model_multihorizon_keras.keras stock_model_multihorizon_keras.keras

# Expose the port (Flask uses 5000 by default)
EXPOSE 5000

# Start the Flask app using gunicorn (recommended for production)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120"]
