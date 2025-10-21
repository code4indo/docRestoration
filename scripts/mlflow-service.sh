#!/usr/bin/env bash
# MLflow Service Startup Script
# This script starts MLflow UI as a service

set -e

# Configuration
PROJECT_DIR="/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration"
USER="lambda_one"
GROUP="lambda_one"
HOST="0.0.0.0"
PORT="5000"
WORKERS=4

# Logging
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/mlflow-service.log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Change to project directory
cd "${PROJECT_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting MLflow UI service..." | tee -a "${LOG_FILE}"
echo "  Host: ${HOST}"
echo "  Port: ${PORT}"
echo "  Workers: ${WORKERS}"
echo "  Project Directory: ${PROJECT_DIR}" | tee -a "${LOG_FILE}"

# Activate virtual environment and start MLflow UI
source .venv/bin/activate

# Start MLflow UI with proper configuration
exec poetry run mlflow ui \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --default-artifact-root "${PROJECT_DIR}/mlruns" \
    2>&1 | tee -a "${LOG_FILE}"