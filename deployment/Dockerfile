FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories
RUN mkdir -p data/processed data/raw models data/eda_outputs \
             data/forecast_outputs data/risk_outputs data/anomaly_outputs

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Default: start API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]