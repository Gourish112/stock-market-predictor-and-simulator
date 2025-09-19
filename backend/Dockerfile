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

# Upgrade pip and install dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ✅ Do NOT force eventlet — remove it
# If eventlet/gevent are in requirements.txt, delete them there too
RUN pip uninstall -y eventlet gevent || true

# Copy project files
COPY . .

# Copy ML model explicitly
COPY stock_model_multihorizon_keras.keras stock_model_multihorizon_keras.keras

# Expose Flask/Gunicorn port
EXPOSE 5000

# Drop to non-root user for security
RUN adduser --disabled-password --gecos '' myuser
USER myuser

# ✅ Use Gunicorn with gthread (built-in threading)
# -w 1   → single worker (so sessions don't break)
# --threads 4 → allow multiple socket clients
CMD ["gunicorn", "app:app", "-k", "gthread", "-w", "1", "--threads", "4", "--bind", "0.0.0.0:5000", "--timeout", "120"]
