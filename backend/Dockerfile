# Use Python 3.10 slim base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt && pip install gunicorn eventlet==0.33.3 redis

# Copy project files
COPY . .

# Copy ML model explicitly (if needed)
COPY stock_model_multihorizon_keras.keras stock_model_multihorizon_keras.keras

# Expose Flask/Gunicorn port
EXPOSE 5000

# Drop to non-root user for security (AFTER installing deps)
RUN adduser --disabled-password --gecos '' myuser
USER myuser

# Start app with gunicorn + eventlet
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]



