#!/bin/bash

# Load environment variables if .env file exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "Loaded environment variables from .env file"
else
  echo "No .env file found, using default environment variables"
fi

# Check if app_secret_key.txt exists
if [ ! -f ./secrets/app_secret_key.txt ]; then
  echo "Creating app_secret_key.txt with a random key"
  mkdir -p ./secrets
  # Generate a random 32-character string for the secret key
  openssl rand -base64 32 | tr -d '\n' > ./secrets/app_secret_key.txt
  chmod 600 ./secrets/app_secret_key.txt
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Build and start production containers
echo "Building production containers..."
docker-compose build

echo "Starting production environment..."
docker-compose up -d

echo "Production environment is now running."
echo "To check container status: docker-compose ps"
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"