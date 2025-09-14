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

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' myuser
USER myuser

# Copy and install Python dependencies
COPY requirements.txt .

# Make sure pip is updated and install requirements, including gunicorn
RUN pip install --upgrade pip
RUN pip install -r requirements.txt gunicorn redis

# Copy your backend code into the container
COPY . .

# Copy your ML model explicitly (if required)
COPY stock_model_multihorizon_keras.keras stock_model_multihorizon_keras.keras

# Expose the port
EXPOSE 5000

# Start the Flask app using gunicorn (recommended for production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]


